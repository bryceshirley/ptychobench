"""
The sample geometries, and the permittivity slices they produce.

A sample is geometry and nothing else: it answers
:meth:`Sample.get_permittivity` for one z and knows nothing about how the
field is propagated. Drawing one lives in :mod:`ptychobench.plotting`, reached
through the thin delegators on :class:`Sample`.

The private builders below are written against the Array API, so a sample built
on a torch grid stays on the device rather than crossing the bus at every
z-step. Two things genuinely cannot: ``scipy.ndimage.gaussian_filter1d`` and
Apoferritin's file load. Those take an explicit, commented host round trip, and
:func:`_as_grid_array` puts the result back where it belongs.
"""

from typing import Any

import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from typing import ClassVar

from .utils import Array, to_numpy


# ==========================================
# Private Helper Functions (Pure Math)
#
# All three builders take the grid as their first argument and read whatever
# they need off it. They used to disagree -- two took a grid, one took the x
# array -- and since only the argument's type distinguished the two
# conventions, a call site passing the wrong one failed at the first attribute
# access rather than at the call. That is how ZigWaveguides shipped broken.
# ==========================================
def _as_grid_array(grid, values: Array) -> Any:
    """
    Return ``values`` as a complex array in the grid's backend and device.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the target backend, dtype and device.
    values : Array
        The geometry to convert.

    Returns
    -------
    Any
        A complex array in the grid's backend.

    Notes
    -----
    The single conversion point for sample geometry. A ``_profile`` may return
    whatever is natural for it -- a device array from the elementwise builders,
    a host array from the two that go through SciPy -- and this makes the
    result uniform before it reaches an operator, which infers its whole
    execution context from the arrays it is handed.
    """
    return grid.xp.asarray(
        values, dtype=grid.complex_dtype, device=grid.resolved_device
    )


def _apply_boundary_mask(grid, eps: Array) -> Any:
    """
    Zero the permittivity in the padded boundary regions.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the transverse coordinates and the domain width.
    eps : Array
        The unmasked profile.

    Returns
    -------
    Any
        A complex array in the grid's backend, free space outside the domain.
    """
    xp = grid.xp
    eps = _as_grid_array(grid, eps)

    # `grid.x` is half-open on [-L/2, +L/2), so this reaches exactly the first
    # column. `where` rather than masked assignment: item assignment by
    # boolean mask is not in the Array API standard.
    boundary_mask = abs(grid.x) >= grid.L / 2

    return xp.where(boundary_mask, xp.zeros_like(eps), eps)


def _polar_perturbation(grid, modulus, phase, profile: Array) -> Any:
    """
    Convert a real ``profile`` and a polar (modulus, phase) into eps.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the backend to build in.
    modulus : float
        Peak magnitude of the perturbation.
    phase : float
        Phase, in radians, at unit profile.
    profile : Array
        The real geometry, in ``[0, 1]``.

    Returns
    -------
    Any
        The complex permittivity perturbation.

    Notes
    -----
    Shared by all three builders, which differ only in how they draw the
    profile. The cast to complex happens before ``exp`` so that a real
    exponential is never taken of what is meant to be a phase.
    """
    xp = grid.xp
    complex_pert = modulus * xp.exp(1j * phase * _as_grid_array(grid, profile))
    return complex_pert * _as_grid_array(grid, profile)


def _build_waveguides(grid, modulus, phase, offset, num_cores, core_width):
    """
    Draw a periodic array of Gaussian waveguide cores.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the transverse coordinates and the domain width.
    modulus : float
        Peak magnitude of the perturbation.
    phase : float
        Phase, in radians, at the core centre.
    offset : float
        Transverse displacement of the whole array, used for the zig-zag.
    num_cores : float
        Cores across the domain; sets the period as ``L / num_cores``.
    core_width : float
        Gaussian width of one core.

    Returns
    -------
    Any
        The complex permittivity perturbation.
    """
    xp = grid.xp

    # Width of the simulation domain
    L = grid.L

    # Determine the period of the waveguides based on the number of cores and domain width
    period = L / num_cores

    x_mod = (grid.x - offset + period / 2) % period - period / 2
    profile = xp.exp(-(x_mod**2) / (core_width**2))

    return _polar_perturbation(grid, modulus, phase, profile)


def _build_sharp_grad_waveguides(grid, modulus, phase, num_cores, core_width, blur=0.0):
    """
    Draw a periodic array of top-hat waveguide cores, optionally blurred.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the transverse coordinates and the domain width.
    modulus : float
        Peak magnitude of the perturbation.
    phase : float
        Phase, in radians, inside a core.
    num_cores : float
        Cores across the domain; sets the period as ``L / num_cores``.
    core_width : float
        Full width of one core.
    blur : float, optional
        Gaussian smoothing sigma, in samples. Zero, the default, leaves the
        edges sharp.

    Returns
    -------
    Any
        The complex permittivity perturbation.
    """
    xp = grid.xp

    # Width of the simulation domain
    L = grid.L

    # Determine the period of the waveguides based on the number of cores and domain width
    period = L / num_cores

    x_mod = (grid.x + period / 2) % period - period / 2
    inside = abs(x_mod) <= core_width / 2
    profile = xp.where(inside, xp.ones_like(x_mod), xp.zeros_like(x_mod))

    if blur > 0.0:
        # Apply Gaussian blur to smooth the edges. SciPy is host-only, so this
        # branch -- and only this branch -- pays a round trip; the profile is
        # O(N) real numbers, against O(N log N) work per propagation step.
        from scipy.ndimage import gaussian_filter1d

        profile = gaussian_filter1d(to_numpy(profile), sigma=blur)

    return _polar_perturbation(grid, modulus, phase, profile)


def _build_balls(
    grid,
    z,
    modulus,
    phase,
    num_balls,
    ball_radius,
    x_period,
    x_amplitude,
    is_zigzag,
):
    """
    Draw a sequence of localised spherical features, by broadcasting.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the transverse coordinates and the propagation distance.
    z : float
        The propagation distance of the slice.
    modulus : float
        Peak magnitude of the perturbation.
    phase : float
        Phase, in radians, at a ball centre.
    num_balls : int
        Balls spread evenly over ``[0, z_prop]``.
    ball_radius : float
        Gaussian radius of one ball.
    x_period : float
        Transverse period the ball centres wrap on.
    x_amplitude : float
        Transverse amplitude of the zig-zag; ignored when it is off.
    is_zigzag : bool
        Displace successive balls in a triangle wave rather than a line.

    Returns
    -------
    Any
        The complex permittivity perturbation for this slice.
    """
    xp = grid.xp
    x_array = grid.x

    # The ball centres and the zig-zag schedule depend only on `num_balls`, not
    # on the grid, so they are built on the host and moved once. `np.piecewise`
    # has no Array API equivalent, and there is nothing to gain from one.
    z_centers = np.linspace(0, grid.z_prop, num_balls)

    if is_zigzag:
        cycle_pos = (np.arange(num_balls) % 8) / 8.0
        x_shift = np.piecewise(
            cycle_pos,
            [
                cycle_pos <= 0.25,
                (cycle_pos > 0.25) & (cycle_pos <= 0.75),
                cycle_pos > 0.75,
            ],
            [
                lambda c: 4.0 * c,
                lambda c: 1.0 - 4.0 * (c - 0.25),
                lambda c: -1.0 + 4.0 * (c - 0.75),
            ],
        )
        x_offsets = x_amplitude * x_shift
    else:
        x_offsets = np.zeros(num_balls)

    def as_column(values: np.ndarray) -> Any:
        """
        Move a host vector to the grid and reshape it to a column.

        Parameters
        ----------
        values : numpy.ndarray
            A length-``num_balls`` host vector.

        Returns
        -------
        Any
            The same values as an ``(num_balls, 1)`` array on the grid.
        """
        return xp.asarray(values, dtype=grid.real_dtype, device=grid.resolved_device)[
            :, None
        ]

    x_mod = (
        x_array[None, :] - as_column(x_offsets) + x_period / 2
    ) % x_period - x_period / 2
    z_diff = z - as_column(z_centers)

    profiles = xp.exp(-(x_mod**2 + z_diff**2) / (ball_radius**2))
    total_profile = xp.sum(profiles, axis=0)

    return _polar_perturbation(grid, modulus, phase, total_profile)


# ==========================================
# Public API (Object-Oriented Samples)
# ==========================================


class Sample(ABC):
    """
    Base class for all physical samples in the simulation.

    Subclasses supply :meth:`_profile` -- the sample's own geometry -- and
    inherit :meth:`get_permittivity`, which applies the free-space boundary to
    it. Splitting the two that way means the boundary rule is written once
    instead of being repeated, correctly, in every subclass.
    """

    def get_permittivity(self, grid, z: float) -> Any:
        """
        Return the 1D complex permittivity array for one z-slice.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates, the domain width, and the
            backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            A complex array of length ``grid.N``, free space outside the
            domain, in ``grid.backend`` on ``grid.device``.
        """
        return _apply_boundary_mask(grid, self._profile(grid, z))

    @abstractmethod
    def _profile(self, grid, z: float) -> Any:
        """
        Return the raw, unmasked permittivity profile for a z-slice.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.

        Notes
        -----
        This is the one method a new sample has to write. The boundary is
        applied by :meth:`get_permittivity`, so implementations should not
        apply it themselves -- nor convert to the grid's backend, which
        :meth:`get_permittivity` also handles. Returning host NumPy is fine.
        """

    # -- plotting delegators ------------------------------------------------
    #
    # The figures live in ptychobench.plotting; a sample is geometry, not a
    # drawing. These three keep the `sample.plot(grid)` spelling the guides and
    # the notebook use. Each still calls show(), which is what they did before,
    # and now also returns the figure so a caller can save or compose it.
    #
    # BenchmarkResult's five delegators do *not* call show(), and the asymmetry
    # is deliberate rather than an oversight. These are the inspect-as-you-go
    # calls: you are at a prompt or in a notebook cell deciding whether the
    # sample you just described is the one you meant, and a figure that does not
    # appear is useless there. A result's plotters are the other case -- they
    # are what generate_report() draws with, nine figures written to disk in one
    # call, and a show() in that path would block the run once per figure. If
    # you want a result's figure on screen, plt.show() it yourself; if you want
    # a sample's figure only on disk, take the returned Figure and save it.
    #
    # Imported inside the methods rather than at module scope: plotting imports
    # nothing from here at runtime, but a top-level import would still pull
    # Matplotlib in for anyone who only wanted a permittivity array. That was
    # the intent all along; a stray `from matplotlib import pyplot` at the top
    # of this module quietly cost it, so `plt` is deferred here too.

    def plot_cross_section_modulus(self, grid, z: float = 0.0, fourier: bool = False):
        """
        Plot |eps(x)| at one z-slice, optionally in the Fourier domain.

        Parameters
        ----------
        grid : SimulationGrid
            The grid to evaluate the sample on.
        z : float, optional
            The propagation distance of the slice. Defaults to ``0.0``.
        fourier : bool, optional
            Transform the slice before plotting. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
            See
            :func:`ptychobench.plotting.plot_sample_cross_section_modulus`.
        """
        from matplotlib import pyplot as plt

        from . import plotting

        fig = plotting.plot_sample_cross_section_modulus(self, grid, z, fourier)
        plt.show()
        return fig

    def plot_cross_section_phase(self, grid, z: float = 0.0):
        """
        Plot the phase of eps(x) at one z-slice.

        Parameters
        ----------
        grid : SimulationGrid
            The grid to evaluate the sample on.
        z : float, optional
            The propagation distance of the slice. Defaults to ``0.0``.

        Returns
        -------
        matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_sample_cross_section_phase`.
        """
        from matplotlib import pyplot as plt

        from . import plotting

        fig = plotting.plot_sample_cross_section_phase(self, grid, z)
        plt.show()
        return fig

    def plot(self, grid):
        """
        Plot the sample's full (x, z) permittivity map, modulus and phase.

        Parameters
        ----------
        grid : SimulationGrid
            The grid to evaluate the sample on.

        Returns
        -------
        matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_sample_profile`.
        """
        from matplotlib import pyplot as plt

        from . import plotting

        fig = plotting.plot_sample_profile(self, grid)
        plt.show()
        return fig

    def max_gradient(self, grid) -> float:
        """
        Measure the largest permittivity gradient over every z-slice.

        Parameters
        ----------
        grid : SimulationGrid
            The grid to evaluate the sample on.

        Returns
        -------
        float
            ``max |d eps / dx|`` over the whole propagation.
        """
        max_grad = 0.0
        for z in grid.z_steps:
            # `np.gradient` has no Array API equivalent, and this is a
            # host-side diagnostic returning a Python float, so bring the
            # slice over rather than hand-rolling a finite difference.
            eps = to_numpy(self.get_permittivity(grid, z))
            grad = float(np.max(np.abs(np.gradient(eps, grid.dx))))
            max_grad = max(max_grad, grad)
        return max_grad


@dataclass
class Apoferritin(Sample):
    """
    Apoferritin biological sample interpolated from a saved .npy 2D cross-section.

    The stored sample is expected to have shape:

        (Nz, Nx)

    where axis 0 is propagation direction z and axis 1 is transverse x.
    """

    modulus: Optional[float] = None
    file_path: Path = Path(__file__).parent / "apoferritin_100Nz400Nx.npy"

    # Class-level cache shared between instances.
    # Keyed by file path, grid size, and modulus.
    _sample_cache: ClassVar[
        dict[tuple[str, int, int, Optional[float]], np.ndarray]
    ] = {}

    def __post_init__(self):
        if self.file_path is None:
            raise ValueError("file_path cannot be None.")

        self.file_path = Path(self.file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {self.file_path}")

    def _cache_key(self, grid) -> tuple[str, int, int, Optional[float]]:
        """
        Build the cache key for this sample, grid and modulus.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the shape the sample is resampled to.

        Returns
        -------
        tuple
            ``(file path, Nz, N, modulus)``.
        """
        return (
            str(self.file_path.resolve()),
            int(grid.Nz),
            int(grid.N),
            self.modulus,
        )

    def _load_raw_sample(self) -> np.ndarray:
        """
        Load the raw apoferritin sample from disk.

        Returns
        -------
        numpy.ndarray
            The stored ``(Nz, Nx)`` real cross-section.

        Raises
        ------
        ValueError
            If the stored array is not two-dimensional.
        """
        data = np.load(str(self.file_path))

        if data.ndim != 2:
            raise ValueError(
                f"Expected apoferritin sample to be 2D, got shape {data.shape}."
            )

        return np.asarray(data, dtype=float)

    def _resample_to_grid(self, data: np.ndarray, grid) -> np.ndarray:
        """
        Resample the sample array to ``(grid.Nz, grid.N)``.

        Parameters
        ----------
        data : numpy.ndarray
            The stored ``(Nz, Nx)`` cross-section.
        grid : SimulationGrid
            Supplies the target shape.

        Returns
        -------
        numpy.ndarray
            The resampled array, or ``data`` itself if it already matched.
        """

        target_shape = (grid.Nz, grid.N)

        if data.shape == target_shape:
            return data

        zoom_z = grid.Nz / data.shape[0]
        zoom_x = grid.N / data.shape[1]

        return zoom(data, (zoom_z, zoom_x), order=1)

    def _apply_modulus(self, data: np.ndarray) -> np.ndarray:
        """
        Optionally rescale the sample to ``[0, modulus]``.

        Parameters
        ----------
        data : numpy.ndarray
            The resampled cross-section.

        Returns
        -------
        numpy.ndarray
            The rescaled array. With ``modulus`` left as None the data is
            unchanged apart from clipping the negative densities, which would
            otherwise read as a phase of pi.
        """

        if self.modulus is None:
            data = np.asarray(data, dtype=float)

            # Remove invalid signed density values.
            # Negative real values would produce phase pi under np.angle.
            data = np.clip(data, 0.0, None)
            return data.astype(np.complex128)

        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min

        if np.isclose(data_range, 0.0):
            return np.zeros_like(data)

        data = (data - data_min) / data_range
        data = data * self.modulus

        return data

    def cache_sample(self, grid) -> np.ndarray:
        """
        Load, resample, rescale and cache the sample for this grid.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the shape the sample is resampled to.

        Returns
        -------
        numpy.ndarray
            The cached ``(grid.Nz, grid.N)`` array.
        """

        key = self._cache_key(grid)

        if key not in self._sample_cache:
            data = self._load_raw_sample()
            data = self._resample_to_grid(data, grid)
            data = self._apply_modulus(data)

            self._sample_cache[key] = data

        return self._sample_cache[key]

    def _profile(self, grid, z: float) -> np.ndarray:
        """
        Return the transverse profile slice nearest to propagation distance z.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the shape the sample is resampled to.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        numpy.ndarray
            The length-``grid.N`` slice, host NumPy.

        Notes
        -----
        The sample comes off disk and is resampled by SciPy, and the cache is
        keyed on the grid's shape rather than its backend so one load serves
        every device. :meth:`Sample.get_permittivity` moves the slice across.
        """

        sample = self.cache_sample(grid)

        # Determine the corresponding index in the sample array based on z
        z_fraction = np.clip(z / grid.z_prop, 0.0, 1.0)
        z_idx = int(round(z_fraction * (sample.shape[0] - 1)))
        # Ensure index is within bounds
        z_idx = np.clip(z_idx, 0, sample.shape[0] - 1)

        return sample[z_idx, :]


@dataclass
class StraightWaveguides(Sample):
    """A periodic array of perfectly straight waveguides."""

    modulus: float = 0.01
    phase: float = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5

    def _profile(self, grid, z: float) -> Any:
        """
        Draw the straight Gaussian cores; ``z`` is unused.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.
        """
        return _build_waveguides(
            grid,
            self.modulus,
            self.phase,
            offset=0.0,
            num_cores=self.num_cores,
            core_width=self.core_width,
        )


@dataclass
class SharpStraightWaveguides(Sample):
    """A periodic array of perfectly straight waveguides."""

    modulus: float = 0.01
    phase: float = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5
    blur: float = 0.0  # Optional Gaussian blur for sharp edges

    def _profile(self, grid, z: float) -> Any:
        """
        Draw the straight top-hat cores; ``z`` is unused.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.
        """
        return _build_sharp_grad_waveguides(
            grid,
            self.modulus,
            self.phase,
            num_cores=self.num_cores,
            core_width=self.core_width,
            blur=self.blur,
        )


@dataclass
class ZigWaveguides(Sample):
    """A periodic array of waveguides that curve (wiggle) along the z-axis."""

    modulus: float = 0.08
    phase: float = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5
    wiggle_amp: float = 2.0

    def _profile(self, grid, z: float) -> Any:
        """
        Draw the cores, displaced sinusoidally with ``z``.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.
        """
        wiggle_offset = self.wiggle_amp * np.sin(2 * np.pi * z / (grid.z_prop / 2))
        return _build_waveguides(
            grid,
            self.modulus,
            self.phase,
            offset=wiggle_offset,
            num_cores=self.num_cores,
            core_width=self.core_width,
        )


@dataclass
class StraightBalls(Sample):
    """A discrete sequence of highly localized spherical features arranged in straight vertical lines."""

    modulus: float = 0.08
    phase: float = 0.0
    num_balls: int = 15
    ball_radius: float = 0.8
    x_period: float = 5.0

    def _profile(self, grid, z: float) -> Any:
        """
        Draw the balls on a straight line.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.
        """
        return _build_balls(
            grid,
            z,
            self.modulus,
            self.phase,
            self.num_balls,
            self.ball_radius,
            self.x_period,
            x_amplitude=0.0,
            is_zigzag=False,
        )


@dataclass
class ZigBalls(Sample):
    """A discrete sequence of highly localized spherical features arranged in a sharp zig-zag pattern."""

    modulus: float = 0.08
    phase: float = 0.0
    num_balls: int = 25
    ball_radius: float = 0.8
    x_period: float = 5.0
    x_amplitude: float = 3.0

    def _profile(self, grid, z: float) -> Any:
        """
        Draw the balls on a zig-zag.

        Parameters
        ----------
        grid : SimulationGrid
            Supplies the transverse coordinates and the backend to build in.
        z : float
            The propagation distance of the slice.

        Returns
        -------
        Any
            The unmasked profile, of length ``grid.N``.
        """
        return _build_balls(
            grid,
            z,
            self.modulus,
            self.phase,
            self.num_balls,
            self.ball_radius,
            self.x_period,
            self.x_amplitude,
            is_zigzag=True,
        )
