"""
The simulation domain.

Its physical parameters, and every array derived from them. Like the rest of the numerical package, the grid is written against the Array
API standard rather than against NumPy, so the field arrays it produces can be
NumPy (the double-precision accuracy reference) or torch, on CPU or GPU. Pick
one with the ``backend`` field; see :mod:`ptychobench.utils` for the resolution
rules and :attr:`SimulationGrid.backend` for what is and is not affected.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from scipy.linalg import dft

from .utils import Backend, resolve_backend, to_numpy


@dataclass
class SimulationGrid:
    """Hold every physical and computational parameter for one simulation."""

    # ---------------------------------------------------------
    # Default Parameters
    # ---------------------------------------------------------
    divergence_angle: float = 0.0
    lam: float = 0.1  # Wavelength of the light (arbitrary units)
    L: float = 100.0  # Total transverse width of the simulation domain
    z_prop: float = 100.0  # Total propagation distance
    N: int = 200  # Number of transverse pixels
    Nz: int = 100  # Number of longitudinal steps (z-steps)
    probe_width: float = 5.0  # Width of the initial Gaussian beam

    #: Which array library the *field* arrays are built with: ``"numpy"``,
    #: ``"torch"`` or ``"cupy"``. A plain string rather than a namespace, so a
    #: grid stays comparable, printable and easy to record alongside a result.
    #: Switch with ``dataclasses.replace(grid, backend="torch")``.
    #:
    #: This governs :attr:`x`, :attr:`kx`, :attr:`propagating_mask`, the two
    #: Fourier symbols and :meth:`get_initial_field` -- everything the
    #: matrix-free path consumes. Two things deliberately stay on the host
    #: whatever it is set to:
    #:
    #: * :attr:`z_steps`, which is a host-side propagation schedule driving a
    #:   Python loop, not a field. Putting it on a GPU would only add a
    #:   synchronising read per step.
    #: * The dense O(N^2) matrices, because the ``"direct"`` integrator hands
    #:   them to ``scipy.linalg.expm``/``sqrtm``, which are host-only. See
    #:   :mod:`ptychobench.numerics.integrators`.
    #:
    #: :func:`ptychobench.benchmark.run_benchmark` runs end to end on any of
    #: them: samples build their permittivity here too, and the histories it
    #: records come back as host NumPy so metrics and plots never see a device
    #: array. ``examples/full_backend_demo.py`` is a complete run on torch;
    #: ``examples/backend_demo.py`` compares the backends against each other.
    backend: str = "numpy"

    #: Which device to build the field arrays on: ``"cpu"``, ``"mps"``,
    #: ``"cuda"``, or ``None`` (the default) for the backend's own choice --
    #: the best accelerator present. Set it explicitly to pin a run to the host
    #: for comparison, which separates a *library* difference from a *device*
    #: one. See :attr:`resolved_device` for what it actually became.
    #:
    #: Precision follows the device, not just the library: torch on CPU is
    #: float64, torch on MPS is float32, because Metal has no FP64.
    device: str | None = None

    # ---------------------------------------------------------
    # The resolved backend
    # ---------------------------------------------------------
    @cached_property
    def _spec(self) -> Backend:
        """
        Resolve the namespace, device and dtypes for this grid, once.

        Returns
        -------
        Backend
            The resolved backend specification.
        """
        return resolve_backend(self.backend, self.device)

    @property
    def xp(self) -> Any:
        """
        The array namespace the field arrays are built in.

        Returns
        -------
        Any
            An Array API namespace.
        """
        return self._spec.xp

    @property
    def resolved_device(self) -> Any:
        """
        The device the field arrays are actually built on.

        Returns
        -------
        Any
            The same as :attr:`device` unless that was left as ``None``, in
            which case whatever the backend chose. ``None`` for backends --
            NumPy among them -- with no concept of placement.
        """
        return self._spec.device

    @property
    def real_dtype(self) -> Any:
        """
        The real floating dtype of this backend and device.

        Returns
        -------
        Any
            A dtype in the backend's own vocabulary.
        """
        return self._spec.real

    @property
    def complex_dtype(self) -> Any:
        """
        The complex dtype matching :attr:`real_dtype` in precision.

        Returns
        -------
        Any
            A dtype in the backend's own vocabulary.
        """
        return self._spec.complex

    # =========================================================
    # The dense DFT matrices
    #
    # Built on first use, not in __post_init__. These are the only O(N^2)
    # objects the grid owns, and only the dense path touches them: a Krylov or
    # split-step run never forms a matrix at all, and used to pay 2 x N^2
    # complex128 -- 128 MB at N = 2048 -- for the privilege of not using them.
    #
    # cached_property, so the cost is paid once per grid rather than once per
    # access. The cache lives in the instance __dict__, which means it is
    # invisible to the dataclass's __repr__ and __eq__, and is dropped if a
    # dimension is reassigned.
    #
    # Host NumPy regardless of `backend`: their only consumer is the "direct"
    # integrator, which hands the assembled matrix to scipy.linalg.expm or
    # sqrtm, and those are host-only. Building them on a GPU would mean copying
    # N^2 numbers back before anything could be done with them.
    # =========================================================
    @cached_property
    def F(self) -> np.ndarray:
        """
        The unitary DFT matrix.

        Returns
        -------
        numpy.ndarray
            ``scipy.linalg.dft(N) / sqrt(N)``, host NumPy.
        """
        return dft(self.N) / np.sqrt(self.N)

    @cached_property
    def F_inv(self) -> np.ndarray:
        """
        The inverse unitary DFT matrix.

        Returns
        -------
        numpy.ndarray
            The conjugate transpose of :attr:`F`, host NumPy.
        """
        return self.F.conj().T

    # =========================================================
    # Derived Properties (Calculated automatically)
    # =========================================================
    @property
    def k0(self) -> float:
        """
        Wavenumber.

        Returns
        -------
        float
            ``2 pi / lam``.
        """
        return 2 * np.pi / self.lam

    @property
    def dx(self) -> float:
        """
        Transverse pixel size.

        Returns
        -------
        float
            ``L / N``.
        """
        return self.L / self.N

    @property
    def dz(self) -> float:
        """
        Longitudinal step size.

        Returns
        -------
        float
            ``z_prop / Nz``.
        """
        return self.z_prop / self.Nz

    @property
    def x(self) -> Any:
        """
        Transverse spatial coordinates, in the grid's backend.

        Returns
        -------
        Any
            An ``(N,)`` real array over the half-open domain ``[-L/2, +L/2)``.

        Notes
        -----
        The last column sits one pixel short of the upper edge. That is what
        makes the DFT periodic, and why :meth:`index_of_x` exists rather than
        being open-coded.
        """
        xp = self.xp
        offsets = xp.arange(self.N, dtype=self.real_dtype, device=self.resolved_device)
        return -self.L / 2 + self.dx * offsets

    @property
    def kx(self) -> Any:
        """
        Transverse spectral (Fourier) coordinates, in the grid's backend.

        Returns
        -------
        Any
            An ``(N,)`` real array in the ``fftfreq`` ordering,
            ``[0, 1, ..., n/2 - 1, -n/2, ..., -1]`` scaled by ``2*pi/(N*dx)``.

        Notes
        -----
        Spelled out with ``arange`` and ``where`` rather than taken from
        ``xp.fft``: ``fft`` is an *optional* extension in the standard, and
        these coordinates are needed even by the dense path.
        """
        xp = self.xp
        index = xp.arange(self.N, dtype=self.real_dtype, device=self.resolved_device)
        # Fold the upper half of the index range onto the negative frequencies.
        wrapped = xp.where(index >= (self.N + 1) // 2, index - self.N, index)
        return (2 * math.pi / (self.N * self.dx)) * wrapped

    @property
    def kx_shifted(self) -> Any:
        """
        :attr:`kx` in ascending order, matching ``fftshift`` output.

        Returns
        -------
        Any
            An ``(N,)`` real array, ascending.

        Notes
        -----
        The abscissa for anything drawn *after* an ``fftshift``, chiefly the
        far-field plots. It exists so that no plotter builds its own axis:
        ``plot_farfield_error`` used to open-code ``linspace(-L/2, L/2, N)``,
        which was the wrong space and, read as an ``x`` axis, a full pixel wide
        of the half-open :attr:`x`.

        Spelled with ``arange`` rather than ``xp.fft.fftshift(self.kx)`` for
        the same reason :attr:`kx` is. Shifting the ``fftfreq`` ordering by
        ``N // 2`` is exactly what ``fftshift`` produces, for even and odd
        ``N`` alike.
        """
        xp = self.xp
        index = xp.arange(self.N, dtype=self.real_dtype, device=self.resolved_device)
        return (2 * math.pi / (self.N * self.dx)) * (index - self.N // 2)

    @property
    def z_steps(self) -> np.ndarray:
        """
        Array of z-coordinates for the propagation loop.

        Returns
        -------
        numpy.ndarray
            An ``(Nz,)`` array over the half-open ``[0, z_prop)``.

        Notes
        -----
        Always NumPy, whatever :attr:`backend` says. This is a host-side
        schedule that drives a Python ``for`` loop and gets handed to
        :mod:`ptychobench.samples` as a scalar; a device array would buy
        nothing and cost a synchronising read per step.
        """
        return np.linspace(0, self.z_prop, self.Nz, endpoint=False)

    def index_of_x(self, x: float) -> int:
        """
        Find the grid column nearest to transverse position ``x``.

        Parameters
        ----------
        x : float
            A transverse position, in the same units as :attr:`L`.

        Returns
        -------
        int
            The column index.

        Raises
        ------
        ValueError
            If ``x`` lies outside the domain. Silently clamping would return a
            plausible-looking edge slice for a position that was never
            simulated.

        Notes
        -----
        The single x -> index mapping in the package. Callers used to roll
        their own, and the two that existed disagreed: one assumed the grid
        spanned ``[-L/2, L/2]`` inclusively, but :attr:`x` is half-open, so
        that version was off by up to a pixel and biased outward.
        """
        if not -self.L / 2 <= x <= self.L / 2:
            raise ValueError(
                f"x = {x} is outside the domain "
                f"[{-self.L / 2}, {self.L / 2}] of this grid"
            )
        # Host arithmetic on a host copy: the answer is a Python int used to
        # slice, so there is nothing to gain from doing the argmin on a GPU.
        return int(np.argmin(np.abs(to_numpy(self.x) - x)))

    # =========================================================
    # Helper Methods (Physics Generation)
    # =========================================================
    def get_initial_field(self) -> Any:
        """
        Build the initial wave function psi_0 at z = 0.

        A Gaussian amplitude with a quadratic phase tilt set by
        :attr:`divergence_angle`.

        Returns
        -------
        Any
            An ``(N,)`` array in the grid's backend, at :attr:`complex_dtype`.

        Notes
        -----
        This is the array whose namespace, device and precision the operators
        then follow, so it is the single place a run's backend is decided in
        practice.
        """
        xp = self.xp
        x = self.x
        amplitude = xp.exp(-(x**2) / (2 * self.probe_width**2))

        # Convert degrees to radians for internal math
        div_angle_rads = self.divergence_angle * math.pi / 180.0

        # Calculate a quadaratic phase shift. The cast happens before exp() so
        # that a real exponential is never taken of what is meant to be a
        # phase: `exp` of a real array is real in every backend.
        phase = div_angle_rads * (x**2) / (2 * self.probe_width)
        phase_shift = xp.exp(1j * xp.astype(phase, self.complex_dtype))

        return xp.astype(amplitude, self.complex_dtype) * phase_shift

    @property
    def propagating_mask(self) -> Any:
        """
        Which Fourier modes propagate.

        Returns
        -------
        Any
            An ``(N,)`` boolean array: True where ``|kx| <= k0``, False for
            the evanescent modes.
        """
        return abs(self.kx) <= self.k0

    @property
    def _kz_sq_over_k0_sq(self) -> Any:
        """
        (kz/k0)^2 with evanescent modes clipped to zero.

        Returns
        -------
        Any
            An ``(N,)`` non-negative real array.

        Notes
        -----
        The single place the band limit is applied. Both symbols are built
        from this, so they cannot drift apart.
        """
        xp = self.xp
        ratio = 1.0 - (self.kx**2) / (self.k0**2)
        # Clip on the sign of `ratio` rather than on `propagating_mask`. The two
        # select the same modes in exact arithmetic -- ratio >= 0 is exactly
        # |kx| <= k0 -- but only this form is self-consistent under rounding. In
        # float32, kx**2 / k0**2 can round a hair above 1 for a mode the mask
        # keeps, and sqrt() of the resulting -1e-8 is NaN. That NaN then travels
        # into the angular spectrum symbol and surfaces much later as an
        # unexplained Arnoldi breakdown.
        #
        # An explicit zeros array rather than the scalar 0.0: `where` is
        # specified for arrays, and backends vary in how forgiving they are.
        return xp.where(ratio > 0.0, ratio, xp.zeros_like(ratio))

    # ---------------------------------------------------------
    # Fourier symbols
    #
    # A "symbol" is the Fourier-space *diagonal* of an operator: the length-N
    # vector g such that the operator is F_inv @ diag(g) @ F. The matrix-free
    # path multiplies by it in O(N) through
    # ptychobench.numerics.transforms.fourier_multiply; the dense path
    # conjugates diag(g) by the DFT to get the O(N^2) matrix below. Both come
    # from the same vector so the two paths cannot disagree about the physics.
    #
    # These are properties rather than get_* methods because, like `kx` and
    # `propagating_mask`, they are O(N) derived vectors. `get_*_operator` is
    # reserved for the O(N^2) matrix builders.
    # ---------------------------------------------------------
    @property
    def kinetic_symbol(self) -> Any:
        """
        Fourier diagonal of the band-limited kinetic operator mu.

        Returns
        -------
        Any
            An ``(N,)`` real array, ``mu(kx) = max(1 - kx^2/k0^2, 0) - 1``.

        Notes
        -----
        Unband-limited, ``mu = F_inv @ diag(-kx^2/k0^2) @ F``; removing the
        evanescent modes sets ``1 + mu = max(1 - kx^2/k0^2, 0)``, so the
        clipped modes take the value -1. Real and even in kx, which is what
        makes the operator self-adjoint.
        """
        return self._kz_sq_over_k0_sq - 1.0

    @property
    def angular_spectrum_symbol(self) -> Any:
        """
        Fourier diagonal of the band-limited angular spectrum operator L.

        Returns
        -------
        Any
            An ``(N,)`` real, even array,
            ``L(kx) = sqrt(max(1 - kx^2/k0^2, 0)) - 1``.

        Notes
        -----
        Evanescent modes are clipped by setting kz = 0, which sends L to -1
        there. The square root is taken of an already non-negative quantity,
        so the branch cut is never approached here -- unlike
        ``sqrt(1 + mu + eps)``, where a potential can push the argument
        negative; see
        :func:`ptychobench.numerics.integrators.warn_if_near_branch_cut`.
        """
        return self.xp.sqrt(self._kz_sq_over_k0_sq) - 1.0

    def get_kinetic_operator(self) -> np.ndarray:
        """
        Assemble the band-limited kinetic operator matrix mu.

        Returns
        -------
        numpy.ndarray
            The ``(N, N)`` dense form of :attr:`kinetic_symbol`, as
            ``F_inv @ diag(mu) @ F``. Host NumPy whatever :attr:`backend` is,
            like :attr:`F`.
        """
        symbol = to_numpy(self.kinetic_symbol).astype(complex)
        return self.F_inv @ np.diag(symbol) @ self.F

    def get_angular_spectrum_operator(self) -> np.ndarray:
        """
        Assemble the band-limited angular spectrum envelope operator L.

        Returns
        -------
        numpy.ndarray
            The ``(N, N)`` dense form of :attr:`angular_spectrum_symbol`, as
            ``F_inv @ diag(L) @ F``. Host NumPy whatever :attr:`backend` is,
            like :attr:`F`.
        """
        symbol = to_numpy(self.angular_spectrum_symbol).astype(complex)
        return self.F_inv @ np.diag(symbol) @ self.F
