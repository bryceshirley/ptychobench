"""
Operator splitting for a separable propagation operator.

Where :mod:`ptychobench.numerics.integrators` handles a general operator by
projecting it onto a Krylov subspace, this module handles the case where the
operator *separates* as ``H = L + N``, with ``L`` diagonal in Fourier space (the
kinetic, or angular-spectrum, part) and ``N`` diagonal in real space (the
potential part). Feit/Fleck [1]_ is exactly this form. Each factor is then
trivial to exponentiate -- elementwise, on its own diagonal -- and a step costs
one FFT pair rather than a matrix function.

The price is that ``L`` and ``N`` do not commute, so ``exp(h(L + N))`` is only
approximated by the product. One ordering is provided: :func:`lie_trotter`,
``exp(hL) exp(hN)``, first order in ``h`` over a fixed distance. A symmetric
Strang product [2]_ used to live here too; it was removed because two splittings
meant the ``"split-step"`` column mixed the splitting error with the operator
error the benchmark exists to measure. See [3]_ for the general theory.

Notes
-----
H3 and H4 add a term that is a *product* across the two bases, so a two-factor
product cannot represent them. They take a third factor ``exp(Omega)``, diagonal
in neither basis and available only as an action, which
:func:`taylor_exponential` evaluates as a truncated series [4]_. That factor is
what makes those two operators available to ``integrator="split-step"``, and
hence to autograd, at all.

:func:`lie_trotter` takes an optional ``aperture``, the anti-alias projection
from :mod:`ptychobench.numerics.apertures`. It rides along in the kinetic
factor's transform pair, so it costs nothing extra, but it is *composed* with
the propagator rather than added to the exponent: a step applies ``P K V``,
which is not ``exp`` of anything. Band limiting therefore changes the operator
being integrated, not merely how it is evaluated.

A Lie-Trotter step *ends* on ``P``, so its output is band limited. That is a
property of this ordering, not of splitting in general -- any product finishing
on a real-space kick repopulates the stopband, and a caller who reorders the
factors has to project again on the way out. Only the field is masked, never the
potential; :mod:`ptychobench.numerics.apertures` carries the measurements and
the comparison with multislice codes.

References
----------
.. [1] M. D. Feit and J. A. Fleck, "Light propagation in graded-index optical
       fibers", Appl. Opt. 17(24), 3990-3998 (1978).
.. [2] G. Strang, "On the construction and comparison of difference schemes",
       SIAM J. Numer. Anal. 5(3), 506-517 (1968).
.. [3] R. I. McLachlan and G. R. W. Quispel, "Splitting methods", Acta
       Numerica 11, 341-434 (2002).
.. [4] Y.-T. Lin and T. F. Duda, "A higher-order split-step Fourier
       parabolic-equation sound propagation solution scheme", J. Acoust. Soc.
       Am. 132(2), EL61-EL67 (2012).
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Any, Callable

from array_api_compat import array_namespace

from .transforms import fourier_multiply
from ..utils import Array, vector_norm


def _exponential_factor(xp: Any, generator: Array, step: complex, dtype: Any) -> Any:
    """
    Build ``exp(step * generator)`` as an elementwise diagonal factor.

    Parameters
    ----------
    xp : Any
        The array namespace to build in.
    generator : Array
        The diagonal of the generator, in its own basis.
    step : complex
        The scalar multiplying it.
    dtype : Any
        The complex working dtype.

    Returns
    -------
    Any
        The factor, in ``dtype``.

    Notes
    -----
    The generator is cast first: ``step`` is imaginary for a propagation, and
    multiplying a complex scalar into a real array relies on promotion rules
    that differ between backends.
    """
    return xp.exp(step * xp.astype(generator, dtype))


@lru_cache(maxsize=8)
def _default_complex_dtype(xp: Any) -> Any:
    """
    Probe the backend's default complex dtype, once per namespace.

    Parameters
    ----------
    xp : Any
        The array namespace to probe.

    Returns
    -------
    Any
        The namespace's default complex dtype.

    Notes
    -----
    The probe is ``asarray(1j)``, which *allocates*; :func:`_working_dtype` runs
    once per step, so an uncached probe is pure overhead against a step that is
    otherwise a handful of kernel launches. Caching on the namespace assumes the
    default dtype does not change mid-run, which is only false if a caller
    reaches for ``torch.set_default_dtype`` inside a propagation loop.
    """
    return xp.asarray(1j).dtype


def _working_dtype(xp: Any, psi: Array) -> Any:
    """
    Pick the complex dtype the factors are built in, at the field's precision.

    Parameters
    ----------
    xp : Any
        The array namespace.
    psi : Array
        The field whose precision the factors match.

    Returns
    -------
    Any
        The complex working dtype.
    """
    return xp.result_type(psi.dtype, _default_complex_dtype(xp))


def _kinetic_step(
    psi: Any,
    kinetic_symbol: Array,
    step: complex,
    aperture: Array | None,
    xp: Any,
    dtype: Any,
) -> Any:
    """
    Apply ``P exp(step * L)`` in one transform pair.

    Parameters
    ----------
    psi : Any
        The complex field in real space.
    kinetic_symbol : Array
        The Fourier diagonal ``L(kx)``.
    step : complex
        The scalar multiplying the generator.
    aperture : Array or None
        The Fourier-space projection ``P``, or None for no band limiting.
    xp : Any
        The array namespace.
    dtype : Any
        The complex working dtype.

    Returns
    -------
    Any
        The advanced field, in real space.
    """
    multiplier = _exponential_factor(xp, kinetic_symbol, step, dtype)
    if aperture is not None:
        # One multiplier, one FFT pair: the projection rides along with the
        # kinetic factor because both are diagonal in the same basis.
        multiplier = xp.astype(aperture, dtype) * multiplier
    return fourier_multiply(psi, multiplier)


def lie_trotter(
    psi: Array,
    kinetic_symbol: Array,
    potential: Array,
    step: complex,
    *,
    aperture: Array | None = None,
) -> Any:
    """
    Advance one first-order Lie-Trotter split step.

    Computes ``exp(step * L) exp(step * N) psi``, optionally with the
    anti-alias projection composed onto the kinetic factor.

    Parameters
    ----------
    psi : Array
        The complex field in real space.
    kinetic_symbol : Array
        The Fourier diagonal ``L(kx)``, in the same bin ordering as
        :attr:`ptychobench.grid.SimulationGrid.kx` -- for example
        :attr:`~ptychobench.grid.SimulationGrid.angular_spectrum_symbol`.
    potential : Array
        The real-space diagonal ``N(x)``, for example ``sqrt(1 + eps) - 1``.
    step : complex
        The scalar multiplying both generators, ``1j * k0 * dz`` for a
        propagation of ``dz``.
    aperture : Array, optional
        A Fourier-space projection applied with the kinetic factor. See the
        module docstring for what the composed action is; in particular it is
        not part of the exponent and does not commute with ``potential``.

    Returns
    -------
    Any
        The advanced field, in the namespace, dtype and device of ``psi``.

    Raises
    ------
    NotImplementedError
        If the array's namespace lacks the optional ``fft`` extension.

    Notes
    -----
    Global error is first order in the step for a fixed total distance,
    with the leading term set by the commutator ``[L, N]``. Exact -- to
    round-off -- whenever either generator vanishes, since the product then
    has only one non-trivial factor.

    The exponentials are recomputed on every call. That is ``O(N)`` against the
    ``O(N log N)`` of the transform, so hoisting them out of a propagation loop
    would complicate the interface to save nothing measurable.
    """
    xp = array_namespace(psi)
    dtype = _working_dtype(xp, psi)

    advanced = _exponential_factor(xp, potential, step, dtype) * psi
    return _kinetic_step(advanced, kinetic_symbol, step, aperture, xp, dtype)


def taylor_exponential(
    apply_generator: Callable[[Any], Any], psi: Array, order: int
) -> Any:
    """
    Apply ``exp(Omega) psi`` by a truncated Taylor series, matrix-free.

    The third factor of a Lin/Duda step. :func:`lie_trotter` exponentiates
    generators that are *diagonal* in some basis, so its exponentials are
    elementwise and exact. This handles a cross term ``Omega``, which is
    diagonal in neither basis and is available only as an action -- so there
    is nothing to exponentiate elementwise and the series is evaluated
    directly.

    Parameters
    ----------
    apply_generator : callable
        Returns ``Omega v`` for a field ``v``. Any scalar prefactor must
        already be folded in: this exponentiates exactly what it is given.
    psi : Array
        The complex field in real space.
    order : int
        Number of terms ``M`` kept after the leading ``1``. Must be at least 1.

    Returns
    -------
    Any
        ``sum_{m=0}^{M} Omega^m psi / m!``, in the namespace, dtype and device
        of ``psi``.

    Raises
    ------
    ValueError
        If ``order`` is less than 1.

    Notes
    -----
    Evaluated by the recurrence ``u_0 = psi``, ``u_m = Omega u_{m-1} / m``,
    which makes ``u_m = Omega^m psi / m!`` directly. Each term costs one
    application of ``Omega`` and no factorial is ever formed -- ``M!``
    overflows a float64 at ``M = 171`` and stops being exact long before that.

    The ``1/m`` belongs in the recurrence *or* in the final sum, never both.
    Lin and Duda's algorithm as usually transcribed states both
    ``u_m = (1/m) Omega u_{m-1}`` and ``psi + sum (1/m!) u_m``, which applies
    the factorial twice and leaves every term from the second on ``1/m!`` too
    small. The form here is the one that reproduces ``exp``, which
    ``test_the_taylor_exponential_matches_a_dense_expm`` checks against
    :func:`scipy.linalg.expm` rather than against the recurrence's own algebra.

    **Truncation costs unitarity, and the series has a horizon.** For an
    anti-Hermitian ``Omega`` -- which is what a propagation cross term is --
    the exact ``exp(Omega)`` is unitary and no truncation of it is, so the norm
    drifts by ``O(r^(M+1))`` per step for ``r = ||Omega||``. That is the whole
    content of "``dz`` sufficiently small": the relative truncation error goes
    like ``r^(M+1) / (M+1)!``, which is negligible for ``r << 1`` and useless
    for ``r`` of order 10 at any ``M`` a propagation loop can afford. Callers
    are responsible for keeping ``r`` small;
    :class:`ptychobench.operators.ForwardOperator` measures it once per
    propagation and warns.
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order!r}")

    term = psi
    total = psi
    for m in range(1, order + 1):
        term = apply_generator(term) / m
        total = total + term
    return total


def _detached(array: Array) -> Any:
    """
    Strip an array of its autograd history, where the backend has one.

    Parameters
    ----------
    array : Array
        The array to detach.

    Returns
    -------
    Any
        The detached array, or ``array`` itself on a backend without gradients.

    Notes
    -----
    The Array API standard has no notion of a gradient, so there is nothing to
    call portably; ``detach`` is torch's spelling.

    Used only by :func:`warn_if_series_truncated`, which reads a Python float
    off the array -- the one operation on the split-step path that would
    otherwise touch the graph. Detaching first says outright that this is a
    diagnostic and not part of the computation.
    """
    detach = getattr(array, "detach", None)
    return detach() if callable(detach) else array


def warn_if_series_truncated(
    generator_psi: Array,
    psi: Array,
    *,
    order: int,
    context: str,
    rtol: float = 1e-6,
) -> bool:
    """
    Warn if :func:`taylor_exponential` is being asked for more than it has.

    The series converges for every ``Omega``, so there is no divergence to
    detect -- the question is whether ``order`` terms are *enough*, and that
    depends entirely on the step size the caller chose. This turns that into a
    number the caller can see rather than a silently wrong field.

    Parameters
    ----------
    generator_psi : Array
        ``Omega psi``, the first term of the series, already computed.
    psi : Array
        The field it was computed from.
    order : int
        The number of terms ``M`` that will be kept.
    context : str
        Named in the warning so the caller can be identified from the message.
    rtol : float, optional
        Estimated relative truncation error above which to warn. Defaults to
        ``1e-6``, which is well below the splitting error of the Lie-Trotter
        product the factor sits inside, so a warning means the series -- and
        not the splitting -- has become the dominant error.

    Returns
    -------
    bool
        True if a :exc:`RuntimeWarning` was issued, False otherwise.

    Notes
    -----
    The size of ``Omega`` is estimated as ``r = ||Omega psi|| / ||psi||``, a
    Rayleigh-quotient-style lower bound on the operator norm rather than the
    norm itself: it can only understate ``r``, so a field that happens to miss
    the largest mode of ``Omega`` will not be warned about even though a later
    step might deserve it. The estimate is free -- the caller has ``Omega psi``
    in hand either way -- where an operator norm would cost a power iteration
    per step.

    The reported error is the first omitted term, ``r^(M+1) / (M+1)!``, which
    bounds the tail whenever ``r < M + 2`` and is the right scale regardless.

    This warns rather than raising, and callers are expected to latch it: the
    ratio is a property of the grid and the step, so it barely moves over a
    propagation, and warning once per step would emit thousands of copies of
    the same message.
    """
    scale = float(vector_norm(_detached(psi)))
    if scale == 0.0:
        return False

    ratio = float(vector_norm(_detached(generator_psi))) / scale
    estimate = ratio ** (order + 1) / math.factorial(order + 1)
    if estimate <= rtol:
        return False

    warnings.warn(
        f"{context}: the Taylor series for exp(Omega) is truncated at "
        f"{order} term(s) but ||Omega psi|| / ||psi|| is {ratio:.3g}, giving a "
        f"relative truncation error of about {estimate:.2e} per step. The "
        "series is also not unitary once truncated, so that error accumulates "
        "as a norm drift. Reduce dz or raise the operator's cross_term_order.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True
