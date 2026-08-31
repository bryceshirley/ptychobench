# src/ptychobench/operators.py
"""
The hierarchy of one-way propagation operators, and how to exponentiate them.

Each operator supplies a generator ``H`` for the envelope equation

    d(psi)/dz = i k0 H psi

and a step advances the field by ``exp(i k0 dz H)``. The classes differ only in
which approximation to ``H = sqrt(1 + mu + eps) - 1`` they implement; how that
exponential is *evaluated* is a separate, orthogonal choice, made with the
``integrator`` argument:

``"direct"``
    Form ``H`` as a dense ``N x N`` matrix and call
    :func:`scipy.linalg.expm`. ``O(N^3)`` work and ``O(N^2)`` storage. This is
    the accuracy reference the other two are tested against, and it is the
    default; it is never routed through the approximations below. Evaluated on
    the host in NumPy double precision on every backend -- SciPy is host-only,
    and a dense matrix is the wrong thing to put on a GPU at these sizes. The
    field is carried across and back, so ``step`` still returns an array in the
    caller's backend; the transfer is what makes ``"direct"`` the slow path on a
    device grid, not merely the ``O(N^3)``. Not differentiable: SciPy detaches.
``"krylov"``
    Project onto a Krylov subspace built from the matrix-free action
    :meth:`ForwardOperator.apply` and exponentiate the small projection. See
    :mod:`ptychobench.numerics.integrators`. Not differentiable either: ``arnoldi``
    reads a Python float for its breakdown test, and the projection goes to
    SciPy.
``"split-step"``
    Exponentiate the Fourier-diagonal and real-space-diagonal halves
    separately, in first-order Lie-Trotter order. See
    :mod:`ptychobench.numerics.splitting`, which explains why that is the only ordering
    on offer: one splitting order keeps this column's integrator error the same
    for every operator in it, so what varies down the column is the operator.
    Available to ``H1`` and ``H2``, whose generators separate, and to ``H3``
    and ``H4``, whose cross terms ride in a third factor evaluated as a
    truncated series -- at the cost of a step that is no longer exactly
    unitary. ``GroundTruthOperator`` has no such decomposition and rejects
    ``"split-step"`` at construction rather than silently dropping terms.
    **The one integrator that carries gradients**: FFTs and elementwise
    products throughout, all in the caller's namespace.

That last point is uneven enough to be worth stating twice. On a torch grid
only ``"split-step"`` is autograd-safe, so it is the only route to a gradient
through a propagation -- which is why H3 and H4 are given a third factor
rather than left to Krylov. The reference is still evaluated densely or on a
Krylov subspace, so the *benchmark* is not differentiable end to end even when
every approximate operator in it is. See the technical notes, "Which
integrators carry gradients", for what that does and does not allow.

The matrix-free paths need the permittivity as a length-``N`` vector, not as
the ``N x N`` diagonal matrix the dense path takes. :meth:`~ForwardOperator.step`
accepts either and extracts the diagonal when given a matrix, so an existing
caller keeps working without paying to build a matrix the fast paths would
immediately discard.

References
----------
.. [1] M. D. Feit and J. A. Fleck, "Light propagation in graded-index optical
       fibers", Appl. Opt. 17(24), 3990-3998 (1978).
.. [2] G. R. Hadley, "Wide-angle beam propagation using Pade approximant
       operators", Opt. Lett. 17(20), 1426-1428 (1992).
.. [3] N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM
       (2008).
.. [4] Lin, Ying-Tsong, and Timothy F. Duda. "A higher-order split-step
       Fourier parabolic-equation sound propagation solution scheme." The
       Journal of the Acoustical Society of America 132.2 (2012): EL61-EL67.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Literal

import numpy as np
from array_api_compat import array_namespace
from scipy.linalg import expm, sqrtm

from .numerics.integrators import krylov_funm, warn_if_near_branch_cut
from .numerics.splitting import (
    lie_trotter,
    taylor_exponential,
    warn_if_series_truncated,
)
from .numerics.transforms import fourier_multiply
from .utils import Array, to_numpy

Integrator = Literal["direct", "krylov", "split-step"]

#: Runtime counterpart of :data:`Integrator`. A Literal is erased at runtime, so
#: the constructor needs its own list to validate against.
INTEGRATORS: tuple[str, ...] = ("direct", "krylov", "split-step")


def _permittivity_vector(E: Array) -> Any:
    """
    Return the permittivity as a length-``N`` vector.

    Parameters
    ----------
    E : Array
        Either the permittivity vector itself, or the ``N x N`` diagonal matrix
        ``diag(eps)`` that the dense path takes.

    Returns
    -------
    Any
        A length-``N`` vector, in ``E``'s namespace, dtype and device. Returned
        unchanged when ``E`` is already one-dimensional.

    Raises
    ------
    ValueError
        If ``E`` is neither a vector nor a square matrix.
    """
    if E.ndim == 1:
        return E
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(
            f"the permittivity must be a length-N vector or an N x N diagonal "
            f"matrix, got shape {tuple(E.shape)}"
        )
    xp = array_namespace(E)
    # Not xp.linalg.diagonal: linalg is an optional Array API extension, and
    # taking a diagonal does not need it.
    index = xp.arange(E.shape[0], device=E.device)
    return E[index, index]


def _as_field_array(psi: Array, values: Any) -> Any:
    """
    Put a grid-side vector into ``psi``'s namespace, dtype and device.

    Parameters
    ----------
    psi : Array
        The field whose namespace, dtype and device the result takes.
    values : Any
        The grid-side vector to carry across.

    Returns
    -------
    Any
        ``values``, ready to multiply ``psi``.

    Notes
    -----
    The grid computes its symbols in NumPy float64. A field may live on another
    backend, and on MPS float64 does not exist at all, so the symbol has to be
    carried across first.
    """
    xp = array_namespace(psi)
    return xp.asarray(values, dtype=psi.dtype, device=psi.device)


def _refractive_potential(eps: Array) -> Any:
    """
    Build ``N(x) = sqrt(1 + eps) - 1``, the real-space part of Feit/Fleck.

    Parameters
    ----------
    eps : Array
        The permittivity as a length-``N`` vector.

    Returns
    -------
    Any
        The real-space diagonal, in ``eps``'s namespace.
    """
    xp = array_namespace(eps)
    return xp.sqrt(1.0 + eps) - 1.0


class ForwardOperator(ABC):
    """
    Base class for the propagation operators.

    Subclasses define the generator ``H``; this class owns the choice of how to
    exponentiate it and the dispatch between the dense and matrix-free paths.

    Parameters
    ----------
    grid : SimulationGrid
        The simulation grid. Supplies the Fourier symbols and the step size.
    integrator : {"direct", "krylov", "split-step"}, optional
        How to evaluate ``exp(i k0 dz H)``. See the module docstring. Defaults
        to ``"direct"``, the accuracy reference.

    Attributes
    ----------
    dz_factor : complex
        ``1j * k0 * dz``, the scalar multiplying the generator in the exponent.

    Raises
    ------
    ValueError
        If ``integrator`` is not one of the three known names.
    NotImplementedError
        If ``"split-step"`` is requested for an operator whose generator does
        not separate.
    """

    #: Human-readable label for plot legends and printed tables, e.g.
    #: ``"Feit/Fleck"``. Declared on the class rather than assigned in
    #: ``__init__`` so that it exists before instantiation and cannot be
    #: omitted silently -- :meth:`__init_subclass__` rejects a subclass that
    #: neither declares nor inherits one. Results are keyed by class name, not
    #: by this, so two operators sharing a label is cosmetic rather than a
    #: corruption of the numbers.
    display_name: ClassVar[str]

    #: Whether the generator splits into one Fourier-diagonal and one
    #: real-space-diagonal term, i.e. is ``L + N`` and nothing else.
    #: Subclasses that set this must also override :meth:`split_step_parts`.
    separable: ClassVar[bool] = False

    #: Whether the generator is ``L + N`` *plus* a cross term that a two-factor
    #: product cannot represent -- H3 and H4. Such an operator is still
    #: reachable by splitting, with :func:`~ptychobench.numerics.splitting.taylor_exponential`
    #: supplying a third factor, so it also gets ``"split-step"``. Subclasses
    #: that set this must override :meth:`split_step_parts` *and*
    #: :meth:`cross_term_action`. Mutually exclusive with :attr:`separable` in
    #: practice, though nothing enforces that.
    cross_term: ClassVar[bool] = False

    #: Terms kept in the Taylor series for ``exp(Omega)``, for an operator with
    #: :attr:`cross_term`. Four is ample at the step sizes this package's own
    #: grids use, where ``||Omega||`` is around 1e-3 and the truncation error is
    #: then far below every other error in the run. It is a class attribute
    #: rather than a constant so a caller taking coarse steps -- where
    #: ``||Omega||`` is not small -- can raise it; :meth:`_step_split` measures
    #: ``||Omega||`` once per propagation and warns when four is not enough.
    cross_term_order: ClassVar[int] = 4

    #: Stopping tolerance for ``integrator="krylov"``, *relative* to the norm
    #: of the field being propagated: :func:`krylov_funm` divides ``||v||`` out
    #: of the a-posteriori residual estimate before comparing against this, so
    #: rescaling the probe does not change the iteration count. The fields here
    #: are unit-normalised, so the distinction rarely bites. Tight enough that
    #: the Krylov error stays well below the modelling error of every operator
    #: here.
    krylov_tol: float = 1e-10

    #: Largest Krylov subspace to build before giving up on the tolerance.
    krylov_max_iter: int = 60

    def __init_subclass__(cls, **kwargs):
        """
        Reject a subclass that has no display name, at import time.

        The old design assigned ``self.name`` in each subclass's ``__init__``,
        so forgetting it produced an operator labelled ``""`` -- discovered, if
        at all, as a blank entry in a plot legend. Checking here turns that
        into an ImportError at the definition.
        """
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "display_name", ""):
            raise TypeError(
                f"{cls.__name__} must declare a class-level `display_name`, "
                "the label used for it in plots and printed tables"
            )

    @property
    def name(self) -> str:
        """
        The operator's display label.

        Returns
        -------
        str
            An alias of :attr:`display_name`.
        """
        return self.display_name

    @classmethod
    def supports_split_step(cls) -> bool:
        """
        Report whether ``integrator="split-step"`` is available.

        Returns
        -------
        bool
            True if the generator separates, or carries a cross term that a
            third factor can absorb.

        Notes
        -----
        A classmethod rather than a property because
        :func:`ptychobench.benchmark.run_benchmark` has to ask before it has an
        instance -- it is deciding which integrator to *construct* the operator
        with. One place answers the question so the constructor's guard and the
        benchmark's fallback cannot drift apart.
        """
        return cls.separable or cls.cross_term

    def __init__(self, grid, integrator: Integrator = "direct"):
        if integrator not in INTEGRATORS:
            raise ValueError(
                f"unknown integrator {integrator!r}; expected one of "
                f"{', '.join(repr(name) for name in INTEGRATORS)}"
            )
        if integrator == "split-step" and not self.supports_split_step():
            # Rejected here rather than at the first step: the mistake is in the
            # construction, and the cross terms cannot be dropped silently.
            raise NotImplementedError(
                f"{type(self).__name__} does not separate into a Fourier-diagonal "
                "and a real-space-diagonal term, and declares no cross term to "
                "carry the remainder, so 'split-step' is unavailable; use "
                "'krylov' or 'direct'"
            )

        self.grid = grid
        self.integrator: Integrator = integrator
        self.dz_factor = 1j * grid.k0 * grid.dz
        # Latched so a propagation loop warns about its configuration once
        # rather than once per step. See _step_split.
        self._checked_cross_term = False

    @cached_property
    def I(self) -> np.ndarray:  # noqa: E743 - `I` is the identity, by convention
        """
        The ``N x N`` identity, host NumPy like the rest of the dense path.

        Returns
        -------
        numpy.ndarray
            The identity matrix.

        Notes
        -----
        Built on first use. It used to be allocated in ``__init__`` for every
        operator, but only :class:`GroundTruthOperator` reads it, and only on the
        dense path -- so a matrix-free run paid N^2 complex128 for something it
        never touched. That is 1 GB at N = 8192, which rather defeats the
        purpose of being matrix-free.
        """
        return np.eye(self.grid.N, dtype=complex)

    @abstractmethod
    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build the generator ``H`` as a dense matrix.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """

    def apply(self, eps: Array, psi: Array) -> Any:
        """
        Apply the generator to a field without forming it.

        The matrix-free counterpart of :meth:`construct_operator`: this must
        return exactly ``construct_operator(diag(eps)) @ psi``, which is what
        ``test_apply_reproduces_the_dense_operator`` checks.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``H psi``, in the namespace, dtype and device of ``psi``.

        Raises
        ------
        NotImplementedError
            If the generator has no cheap elementwise action, as for
            :class:`GroundTruthOperator`, whose ``H`` is a matrix square root.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no matrix-free action for its generator"
        )

    def split_step_parts(self, eps: Array) -> tuple[Any, Any]:
        """
        Split the generator into ``(kinetic_symbol, potential)``.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.

        Returns
        -------
        tuple of Any
            The Fourier diagonal and the real-space diagonal, in the argument
            order :func:`ptychobench.numerics.splitting.lie_trotter` takes. Their sum is
            the whole generator for a :attr:`separable` operator, and the whole
            generator *less* :meth:`cross_term_action` for one with a
            :attr:`cross_term`.

        Raises
        ------
        NotImplementedError
            If the generator does not separate. Unreachable through
            :meth:`step`, which rejects ``"split-step"`` at construction, but
            stated here so the contract is visible on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not separate into a Fourier-diagonal "
            "and a real-space-diagonal term"
        )

    def cross_term_action(self, eps: Array, psi: Array) -> Any:
        """
        The generator's cross term applied to a field, ``C psi``.

        The part of ``H`` that :meth:`split_step_parts` leaves out, so that

            H psi = L psi + N psi + cross_term_action(eps, psi)

        for an operator declaring a :attr:`cross_term`. Sign included: for H4
        this returns ``-(L N + N L) psi / 2``, not the bare product.

        This exists so the two paths cannot disagree. :meth:`apply` -- the
        Krylov path, and what
        ``test_apply_reproduces_the_dense_operator`` pins against the dense
        matrix -- is written in terms of it, and so is the split-step path
        below, so a sign slip or a transposed factor cannot be correct in one
        and wrong in the other.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``C psi``, in the namespace, dtype and device of ``psi``.

        Raises
        ------
        NotImplementedError
            If the generator has no cross term, which is the default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no cross term beyond L + N"
        )

    def _cross_term_generator(self, eps: Array, psi: Array) -> Any:
        """
        Build ``Omega psi``, the exponent of the split step's third factor.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``Omega psi`` for ``Omega = dz_factor * C``.

        Notes
        -----
        The scalar is folded in here so that
        :func:`~ptychobench.numerics.splitting.taylor_exponential`
        exponentiates exactly what it is handed. Both callers in
        :meth:`_step_split` go through this rather than scaling
        :meth:`cross_term_action` themselves, so the truncation diagnostic and
        the series it is diagnosing cannot disagree about what ``Omega`` is.
        """
        return self.dz_factor * self.cross_term_action(eps, psi)

    def step(self, E: Array, psi: Array) -> Any:
        """
        Advance the field by one longitudinal step.

        Parameters
        ----------
        E : Array
            The permittivity, either as a length-``N`` vector or as the
            ``N x N`` diagonal matrix ``diag(eps)``. The dense path needs the
            matrix and builds it if given a vector; the matrix-free paths need
            the vector and extract it if given a matrix.
        psi : Array
            The field to advance. The dense path also accepts an ``N x N``
            matrix here, which propagates every column at once and so yields
            the propagator itself when handed the identity; the matrix-free
            paths take a field.

        Returns
        -------
        Any
            The field after ``dz``, in the namespace, dtype and device of
            ``psi`` -- including on ``"direct"``, which computes on the host
            and carries the result back.
        """
        if self.integrator == "direct":
            # The dense path is host NumPy whatever backend the field came from:
            # expm and sqrtm are SciPy, and an N x N matrix has no business on a
            # GPU at these sizes anyway. The round trip is explicit because
            # np.asarray quietly succeeds for a CPU tensor and raises for an MPS
            # one -- so the default integrator used to work on torch:cpu (while
            # silently handing back a NumPy array) and crash on torch:mps.
            dense = to_numpy(E)
            if dense.ndim == 1:
                dense = np.diag(dense.astype(complex))
            stepped = expm(self.dz_factor * self.construct_operator(dense)).dot(
                to_numpy(psi)
            )
            # Back to the caller's backend, so a field does not change type
            # depending on which integrator advanced it.
            return _as_field_array(psi, stepped)

        eps = _permittivity_vector(E)
        if self.integrator == "krylov":
            return self._step_krylov(eps, psi)
        return self._step_split(eps, psi)

    def _step_krylov(self, eps: Array, psi: Array) -> Any:
        """
        Exponentiate the generator on a Krylov subspace built from ``apply``.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The field to advance.

        Returns
        -------
        Any
            The field after ``dz``.
        """

        def scaled_generator(x: Any) -> Any:
            """
            Apply ``dz_factor * H`` to a field.

            Parameters
            ----------
            x : Any
                The field to apply the generator to.

            Returns
            -------
            Any
                ``dz_factor * H x``.
            """
            return self.dz_factor * self.apply(eps, x)

        return krylov_funm(
            scaled_generator,
            psi,
            expm,
            max_iter=self.krylov_max_iter,
            tol=self.krylov_tol,
        )

    def _step_split(self, eps: Array, psi: Array) -> Any:
        """
        Exponentiate the generator as a product of diagonal factors.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The field to advance.

        Returns
        -------
        Any
            The field after ``dz``.

        Warns
        -----
        RuntimeWarning
            If the Taylor series for ``exp(Omega)`` looks unconverged at
            :attr:`cross_term_order` terms, which means the step is too large
            for the method. Checked once per operator, not once per step.

        Notes
        -----
        ``exp(dz_factor * L) exp(dz_factor * N) psi`` for a :attr:`separable`
        operator. For one with a :attr:`cross_term`, a third factor
        ``exp(Omega)`` goes on the inside, evaluated by a truncated series:

            psi -> exp(dz_factor L) exp(dz_factor N) exp(Omega) psi

        The whole path is FFTs and elementwise products in the caller's
        namespace, so it is the one integrator that carries gradients -- which
        is the point of routing H3 and H4 through it rather than leaving them
        to Krylov.
        """
        kinetic, potential = self.split_step_parts(eps)
        if self.cross_term:
            if not self._checked_cross_term:
                # Latched whether or not it warned. The ratio is set by the grid
                # and the step, so it barely moves over a propagation, and the
                # check costs a whole extra application of Omega -- two FFT
                # pairs -- which is not worth paying every step to re-learn the
                # same answer.
                self._checked_cross_term = True
                warn_if_series_truncated(
                    self._cross_term_generator(eps, psi),
                    psi,
                    order=self.cross_term_order,
                    context=f"{type(self).__name__} cross term",
                )
            psi = taylor_exponential(
                lambda field: self._cross_term_generator(eps, field),
                psi,
                self.cross_term_order,
            )
        return lie_trotter(
            psi,
            _as_field_array(psi, kinetic),
            _as_field_array(psi, potential),
            self.dz_factor,
        )


class GroundTruthOperator(ForwardOperator):
    """
    The exact one-way generator, ``H = sqrt(1 + mu + eps) - 1``.

    The reference every approximation in this module is measured against. It
    exists because the transverse problem is one-dimensional, which is what
    makes forming the ``N x N`` square root affordable at all.

    Parameters
    ----------
    grid : SimulationGrid
        The simulation grid.
    integrator : {"direct", "krylov"}, optional
        How to evaluate the exponential. ``"split-step"`` is rejected: a matrix
        square root has no ``L + N`` decomposition. Defaults to ``"direct"``.
    """

    display_name: ClassVar[str] = "Ground Truth"

    def __init__(self, grid, integrator: Integrator = "direct"):
        super().__init__(grid, integrator)
        # Latched so a propagation loop warns about its configuration once,
        # rather than once per Arnoldi iteration of every step.
        self._warned_branch_cut = False

    @cached_property
    def I_plus_M(self) -> np.ndarray:
        """
        ``1 + mu`` as a dense matrix, the constant part of the root's argument.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.

        Notes
        -----
        Lazy for the same reason as :attr:`ForwardOperator.I`, and it matters
        more here: building it also forces the grid's DFT matrices, so an eager
        version made a Krylov run allocate three N x N matrices before taking
        its first step.
        """
        return self.I + self.grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build ``H = sqrt(1 + mu + eps) - 1`` as a dense matrix square root.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """
        return sqrtm(self.I_plus_M + E) - self.I

    def _step_krylov(self, eps: Array, psi: Array) -> Any:
        """
        Project ``A = 1 + mu + eps`` and apply a composed function to it.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The field to advance.

        Returns
        -------
        Any
            The field after ``dz``.

        Warns
        -----
        RuntimeWarning
            If the projection has eigenvalues near the negative real axis. The
            principal square root is ill-conditioned there, and the dense and
            matrix-free paths then disagree by far more than the Krylov
            tolerance -- not because either has failed to converge, but because
            the quantity they are both computing is not well determined.

        Notes
        -----
        The generator is a matrix square root, so it has no elementwise action
        and :meth:`apply` is unavailable. What *is* matrix-free is the argument
        of the root, ``A``: one Fourier multiply plus one real-space multiply.
        Projecting ``A`` and evaluating

            f(z) = exp(dz_factor * (sqrt(z) - 1))

        on the small projection covers the root and the exponential in a single
        Arnoldi run, rather than nesting a Krylov solve for the root inside a
        Krylov solve for the exponential.
        """
        mu = _as_field_array(psi, self.grid.kinetic_symbol)

        def helmholtz(x: Any) -> Any:
            """
            Apply ``A = 1 + mu + eps`` to a field, matrix-free.

            Parameters
            ----------
            x : Any
                The field to apply ``A`` to.

            Returns
            -------
            Any
                ``A x``.
            """
            return x + fourier_multiply(x, mu) + eps * x

        def propagator(matrix: np.ndarray) -> np.ndarray:
            """
            Evaluate ``exp(dz_factor * (sqrt(z) - 1))`` on the projection.

            Parameters
            ----------
            matrix : numpy.ndarray
                The small Hessenberg projection of ``A``.

            Returns
            -------
            numpy.ndarray
                The function of the projection.
            """
            # A permittivity that drives an eigenvalue of A negative puts the
            # root on its branch cut, where the principal value jumps. That is
            # physics -- a mode turning evanescent -- but it is worth saying out
            # loud rather than quietly returning a number.
            if not self._warned_branch_cut and warn_if_near_branch_cut(
                matrix, context="sqrt(1 + mu + eps)"
            ):
                self._warned_branch_cut = True
            root = sqrtm(matrix)
            return expm(self.dz_factor * (root - np.eye(matrix.shape[0])))

        return krylov_funm(
            helmholtz,
            psi,
            propagator,
            max_iter=self.krylov_max_iter,
            tol=self.krylov_tol,
        )


class ParaxialOperator(ForwardOperator):
    """
    H1, the paraxial approximation ``H = (eps + mu) / 2``.

    Both terms are truncated at first order, so this is wrong even in free
    space: at ``eps = 0`` it gives ``mu / 2`` where the exact answer is
    ``sqrt(1 + mu) - 1``.
    """

    display_name: ClassVar[str] = "Paraxial"

    separable: ClassVar[bool] = True

    @cached_property
    def half_M(self) -> np.ndarray:
        """
        ``mu / 2`` as a dense matrix. Lazy; see :attr:`ForwardOperator.I`.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.
        """
        return 0.5 * self.grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build ``H = (eps + mu) / 2`` as a dense matrix.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """
        return 0.5 * E + self.half_M

    def apply(self, eps: Array, psi: Array) -> Any:
        """
        Apply ``H = (eps + mu) / 2`` without forming it.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``(eps psi + mu psi) / 2``, in the namespace, dtype and device of ``psi``.
        """
        mu = _as_field_array(psi, self.grid.kinetic_symbol)
        return 0.5 * (eps * psi + fourier_multiply(psi, mu))

    def split_step_parts(self, eps: Array) -> tuple[Any, Any]:
        """
        Split ``H`` into ``(mu / 2, eps / 2)``.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.

        Returns
        -------
        tuple of Any
            The Fourier diagonal and the real-space diagonal.
        """
        return 0.5 * self.grid.kinetic_symbol, 0.5 * eps


class FeitFleckOperator(ForwardOperator):
    """
    H2, the Feit/Fleck form ``H = L + N``.

    ``L = sqrt(1 + mu) - 1`` is exact in the kinetic term alone and
    ``N = sqrt(1 + eps) - 1`` exact in the potential alone; what is dropped is
    every cross term between them, the first of which is ``-(eps mu + mu eps)/8``.
    This is the one operator in the hierarchy that is *by construction* a sum of
    a Fourier-diagonal and a real-space-diagonal term, which is what makes
    split-step propagation possible at all.
    """

    display_name: ClassVar[str] = "Feit/Fleck"

    separable: ClassVar[bool] = True

    @cached_property
    def L_op(self) -> np.ndarray:
        """
        ``L`` as a dense matrix. Lazy; see :attr:`ForwardOperator.I`.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.
        """
        return self.grid.get_angular_spectrum_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build ``H = L + N`` as a dense matrix.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """
        eps = np.diag(E)
        N_op = np.diag(np.sqrt(1 + eps) - 1.0)

        return self.L_op + N_op

    def apply(self, eps: Array, psi: Array) -> Any:
        """
        Apply ``H = L + N`` without forming it.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``L psi + N psi``, in the namespace, dtype and device of ``psi``.
        """
        symbol = _as_field_array(psi, self.grid.angular_spectrum_symbol)
        return fourier_multiply(psi, symbol) + _refractive_potential(eps) * psi

    def split_step_parts(self, eps: Array) -> tuple[Any, Any]:
        """
        Split ``H`` into ``(L, N)``, which is the whole generator.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.

        Returns
        -------
        tuple of Any
            The Fourier diagonal and the real-space diagonal.
        """
        return self.grid.angular_spectrum_symbol, _refractive_potential(eps)


class YevickThomsonOperator(ForwardOperator):
    """
    H3, Yevick/Thomson: ``H2`` with the leading cross term restored.

    ``H = L + N - (eps mu + mu eps) / 8``. The symmetrised product is what makes
    the generator self-adjoint for a real permittivity, and hence the step
    unitary; the unsymmetrised ``-eps mu / 4`` would match the same Taylor
    expansion and quietly lose norm conservation.

    The cross term is a product of a Fourier-diagonal and a real-space-diagonal
    factor, so it is diagonal in neither basis and no two-factor product can
    represent it. Split-step propagation is still available, with the cross term
    carried by a third factor evaluated as a truncated series; see
    :meth:`ForwardOperator._step_split`.
    """

    display_name: ClassVar[str] = "Yevick/Thomson"

    cross_term: ClassVar[bool] = True

    @cached_property
    def L_op(self) -> np.ndarray:
        """
        ``L`` as a dense matrix. Lazy; see :attr:`ForwardOperator.I`.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.
        """
        return self.grid.get_angular_spectrum_operator()

    @cached_property
    def mu_op(self) -> np.ndarray:
        """
        ``mu`` as a dense matrix. Lazy; see :attr:`ForwardOperator.I`.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.
        """
        return self.grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build ``H = L + N - (eps mu + mu eps) / 8`` as a dense matrix.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """
        eps_vec = np.diag(E)
        # np.diag(f(eps_vec)) rather than f(E): elementwise on the full matrix
        # happens to give the same answer here, because the off-diagonal zeros
        # go to sqrt(1) - 1 = 0, but only for this f and only while E is exactly
        # diagonal. Spelled the way the two sibling operators spell it.
        N_op = np.diag(np.sqrt(1 + eps_vec) - 1.0)
        H2 = self.L_op + N_op
        cross_term = (1 / 8) * (
            (self.mu_op * eps_vec) + (self.mu_op * eps_vec[:, None])
        )
        return H2 - cross_term

    def apply(self, eps: Array, psi: Array) -> Any:
        """
        Apply ``H = L + N - (eps mu + mu eps) / 8`` without forming it.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``H psi``, in the namespace, dtype and device of ``psi``.
        """
        symbol = _as_field_array(psi, self.grid.angular_spectrum_symbol)
        h2 = fourier_multiply(psi, symbol) + _refractive_potential(eps) * psi
        return h2 + self.cross_term_action(eps, psi)

    def split_step_parts(self, eps: Array) -> tuple[Any, Any]:
        """
        Split out the ``L + N`` part, which is Feit/Fleck's exactly.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.

        Returns
        -------
        tuple of Any
            The Fourier diagonal and the real-space diagonal.

        Notes
        -----
        H3 *is* H2 plus a cross term, so the two diagonal factors are shared
        and the whole difference between the operators lives in
        :meth:`cross_term_action`.
        """
        return self.grid.angular_spectrum_symbol, _refractive_potential(eps)

    def cross_term_action(self, eps: Array, psi: Array) -> Any:
        """
        Apply the cross term ``-(eps mu + mu eps) / 8``.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``-(eps mu + mu eps) psi / 8``, in the namespace, dtype and device of ``psi``.
        """
        mu = _as_field_array(psi, self.grid.kinetic_symbol)
        # Each factor is applied in the basis it is diagonal in, innermost
        # first: mu(eps psi) for the mu-eps ordering, eps(mu psi) for eps-mu.
        cross = fourier_multiply(eps * psi, mu) + eps * fourier_multiply(psi, mu)
        return -cross / 8.0


class LinDudaOperator(ForwardOperator):
    """
    H4, Lin/Duda: ``H2`` with the cross term taken between ``L`` and ``N``.

    ``H = L + N - (L N + N L) / 2``. Same order of accuracy as H3, but the
    correction is built from the resummed factors rather than from the raw
    ``eps`` and ``mu``, so it stays better behaved at wide angles.

    Like H3, the cross term is a product across the two bases, and like H3 it
    reaches ``"split-step"`` through a third, series-evaluated factor rather
    than through the two-factor product. That is the form Lin and Duda give.
    """

    display_name: ClassVar[str] = "Lin/Duda"

    cross_term: ClassVar[bool] = True

    @cached_property
    def L_op(self) -> np.ndarray:
        """
        ``L`` as a dense matrix. Lazy; see :attr:`ForwardOperator.I`.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` matrix.
        """
        return self.grid.get_angular_spectrum_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Build ``H = L + N - (L N + N L) / 2`` as a dense matrix.

        Parameters
        ----------
        E : numpy.ndarray
            The ``N x N`` diagonal permittivity matrix ``diag(eps)``.

        Returns
        -------
        numpy.ndarray
            The ``N x N`` generator.
        """
        eps = np.diag(E)

        # Keep 1D vector for optimized array broadcasting
        n_vec = np.sqrt(1 + eps) - 1.0
        N_op = np.diag(n_vec)

        H2 = self.L_op + N_op

        # Fast O(N^2) cross-term calculation using broadcasting
        cross_term = 0.5 * ((self.L_op * n_vec) + (self.L_op * n_vec[:, None]))

        H4 = H2 - cross_term
        return H4

    def apply(self, eps: Array, psi: Array) -> Any:
        """
        Apply ``H = L + N - (L N + N L) / 2`` without forming it.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``H psi``, in the namespace, dtype and device of ``psi``.
        """
        symbol = _as_field_array(psi, self.grid.angular_spectrum_symbol)
        h2 = fourier_multiply(psi, symbol) + _refractive_potential(eps) * psi
        return h2 + self.cross_term_action(eps, psi)

    def split_step_parts(self, eps: Array) -> tuple[Any, Any]:
        """
        Split out the ``L + N`` part, which is Feit/Fleck's exactly. See H3.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.

        Returns
        -------
        tuple of Any
            The Fourier diagonal and the real-space diagonal.
        """
        return self.grid.angular_spectrum_symbol, _refractive_potential(eps)

    def cross_term_action(self, eps: Array, psi: Array) -> Any:
        """
        Apply the cross term ``-(L N + N L) / 2``.

        Parameters
        ----------
        eps : Array
            The permittivity as a length-``N`` vector.
        psi : Array
            The complex field in real space.

        Returns
        -------
        Any
            ``-(L N + N L) psi / 2``, in the namespace, dtype and device of ``psi``.
        """
        symbol = _as_field_array(psi, self.grid.angular_spectrum_symbol)
        n_vec = _refractive_potential(eps)
        cross = fourier_multiply(n_vec * psi, symbol) + n_vec * fourier_multiply(
            psi, symbol
        )
        return -0.5 * cross
