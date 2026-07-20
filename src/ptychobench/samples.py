import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from matplotlib import pyplot as plt
from typing import ClassVar


# ==========================================
# Private Helper Functions (Pure Math)
# ==========================================
def _apply_boundary_mask(eps: np.ndarray, x_array: np.ndarray, L: float) -> np.ndarray:
    """
    Free space in the padded boundary regions.

    Returns a complex copy so downstream propagation code has consistent dtype.
    """

    eps = np.asarray(eps, dtype=np.complex128).copy()

    boundary_mask = np.abs(x_array) >= L / 2

    eps[boundary_mask] = 0.0 + 0.0j

    return eps


def _build_waveguides(grid, modulus, phase, offset, num_cores, core_width):
    """
    Core logic for generating periodic waveguides using modulus and phase.
    """
    # Width of the simulation domain
    L = grid.L

    # Determine the period of the waveguides based on the number of cores and domain width
    period = L / num_cores

    x_mod = (grid.x - offset + period / 2) % period - period / 2
    profile = np.exp(-(x_mod**2) / (core_width**2))

    # Convert polar (modulus, phase) to a complex perturbation
    complex_pert = modulus * np.exp(1j * phase * profile)
    return complex_pert * profile


def _build_sharp_grad_waveguides(grid, modulus, phase, num_cores, core_width, blur=0.0):
    """
    Builds a periodic array of waveguides with sharp edges, optionally blurred by a Gaussian kernel.
    """
    # Width of the simulation domain
    L = grid.L

    # Determine the period of the waveguides based on the number of cores and domain width
    period = L / num_cores

    x_mod = (grid.x + period / 2) % period - period / 2
    profile = np.where(np.abs(x_mod) <= core_width / 2, 1.0, 0.0)

    if blur > 0.0:
        # Apply Gaussian blur to smooth the edges
        from scipy.ndimage import gaussian_filter1d

        profile = gaussian_filter1d(profile, sigma=blur)

    # Convert polar (modulus, phase) to a complex perturbation
    complex_pert = modulus * np.exp(1j * phase * profile)
    return complex_pert * profile


def _build_balls(
    x_array,
    z,
    modulus,
    phase,
    z_prop,
    num_balls,
    ball_radius,
    x_period,
    x_amplitude,
    is_zigzag,
):
    """Core logic for generating localized spherical features using array broadcasting."""
    z_centers = np.linspace(0, z_prop, num_balls)

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

    x_mod = (
        x_array[None, :] - x_offsets[:, None] + x_period / 2
    ) % x_period - x_period / 2
    z_diff = z - z_centers[:, None]

    profiles = np.exp(-(x_mod**2 + z_diff**2) / (ball_radius**2))
    total_profile = np.sum(profiles, axis=0)

    # Convert polar (modulus, phase) to a complex perturbation
    complex_pert = modulus * np.exp(1j * phase * total_profile)
    return complex_pert * total_profile


# ==========================================
# Public API (Object-Oriented Samples)
# ==========================================


class Sample(ABC):
    """Base class for all physical samples in the simulation."""

    @abstractmethod
    def get_permittivity(self, grid, z: float) -> np.ndarray:
        """Returns the 1D complex permittivity array for a specific z-slice."""
        pass

    def plot_cross_section_modulus(self, grid, z: float = 0.0, fourier: bool = False):
        """
        Plots the modulus and phase of the permittivity at a specific z-slice.

        Parameters:
            grid: The simulation grid object.
            z (float): The propagation distance at which to evaluate the permittivity.
            fourier (bool): If True, plots the Fourier transform of the permittivity.
        """
        eps = self.get_permittivity(grid, z)
        if fourier:
            eps = np.fft.fft(eps)
            eps = np.fft.fftshift(eps)
            title = f"Modulus of Permittivity at z={z:.2f} (Fourier Domain)"
        else:
            title = f"Modulus of Permittivity at z={z:.2f} (Spatial Domain)"
        plt.plot(grid.x, np.abs(eps))
        plt.xlabel("Transverse coordinate x")
        plt.ylabel("|ε(x, z)|")
        plt.title(title)
        plt.show()
        plt.close()

    def plot_cross_section_phase(self, grid, z: float = 0.0):
        """Plots the phase of the permittivity at a specific z-slice."""
        eps = self.get_permittivity(grid, z)
        plt.plot(grid.x, np.angle(eps))
        plt.xlabel("Transverse coordinate x")
        plt.ylabel("∠ε(x, z)")
        plt.title(f"Phase of Permittivity at z={z:.2f}")
        plt.show()

    def max_gradient(self, grid) -> float:
        """
        Computes the maximum gradient of the permittivity across all z-slices.
        """
        max_grad = 0.0
        for z in grid.z_steps:
            eps = self.get_permittivity(grid, z)
            grad = np.max(np.abs(np.gradient(eps, grid.dx)))
            max_grad = max(max_grad, grad)
        return max_grad

    def plot(self, grid):
        """Compute and plot modulus and phase of the sample."""
        Nz = grid.Nz
        Nx = grid.N

        # Build sample on-the-fly
        sample_history = np.zeros((Nz, Nx), dtype=np.complex128)

        for i, z in enumerate(grid.z_steps):
            sample_history[i, :] = self.get_permittivity(grid, z)

        extent = [
            grid.x[0],
            grid.x[-1],
            grid.z_steps[0],
            grid.z_steps[-1],
        ]

        fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Modulus
        im_real = ax_real.imshow(
            np.abs(sample_history),
            extent=extent,
            aspect="auto",
            cmap="viridis",
            origin="upper",
        )
        ax_real.set_title("Modulus of Sample")
        ax_real.set_xlabel("Transverse coordinate x")
        ax_real.set_ylabel("Propagation distance z")
        fig.colorbar(im_real, ax=ax_real)

        # Phase
        im_imag = ax_imag.imshow(
            np.unwrap(np.angle(sample_history), axis=1),
            extent=extent,
            aspect="auto",
            cmap="plasma",
            origin="upper",
        )
        ax_imag.set_title("Phase of Sample (radians)")
        ax_imag.set_xlabel("Transverse coordinate x")
        fig.colorbar(im_imag, ax=ax_imag)

        fig.suptitle("Refractive Index Environment")

        plt.tight_layout()
        plt.show()


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
        Builds the cache key for this sample/grid/modulus combination.
        """
        return (
            str(self.file_path.resolve()),
            int(grid.Nz),
            int(grid.N),
            self.modulus,
        )

    def _load_raw_sample(self) -> np.ndarray:
        """
        Loads the raw apoferritin sample from disk.
        """
        data = np.load(str(self.file_path))

        if data.ndim != 2:
            raise ValueError(
                f"Expected apoferritin sample to be 2D, got shape {data.shape}."
            )

        return np.asarray(data, dtype=float)

    def _resample_to_grid(self, data: np.ndarray, grid) -> np.ndarray:
        """
        Resamples the sample array to match the simulation grid shape:

            target shape = (grid.Nz, grid.N)
        """

        target_shape = (grid.Nz, grid.N)

        if data.shape == target_shape:
            return data

        zoom_z = grid.Nz / data.shape[0]
        zoom_x = grid.N / data.shape[1]

        return zoom(data, (zoom_z, zoom_x), order=1)

    def _apply_modulus(self, data: np.ndarray) -> np.ndarray:
        """
        Optionally rescales the sample to [0, modulus].

        If modulus is None, the raw/interpolated data is left unchanged.
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
        Loads, resamples, rescales, and caches the sample for this grid.
        """

        key = self._cache_key(grid)

        if key not in self._sample_cache:
            data = self._load_raw_sample()
            data = self._resample_to_grid(data, grid)
            data = self._apply_modulus(data)

            self._sample_cache[key] = data

        return self._sample_cache[key]

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        """
        Returns the transverse permittivity/profile slice at propagation position z.
        """

        sample = self.cache_sample(grid)

        # Determine the corresponding index in the sample array based on z
        z_fraction = np.clip(z / grid.z_prop, 0.0, 1.0)
        z_idx = int(round(z_fraction * (sample.shape[0] - 1)))
        # Ensure index is within bounds
        z_idx = np.clip(z_idx, 0, sample.shape[0] - 1)

        eps = sample[z_idx, :]

        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class StraightWaveguides(Sample):
    """A periodic array of perfectly straight waveguides."""

    modulus: float = 0.01
    phase: Optional[float] = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        eps = _build_waveguides(
            grid,
            self.modulus,
            self.phase,
            offset=0.0,
            num_cores=self.num_cores,
            core_width=self.core_width,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class SharpStraightWaveguides(Sample):
    """A periodic array of perfectly straight waveguides."""

    modulus: float = 0.01
    phase: Optional[float] = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5
    blur: float = 0.0  # Optional Gaussian blur for sharp edges

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        eps = _build_sharp_grad_waveguides(
            grid,
            self.modulus,
            self.phase,
            num_cores=self.num_cores,
            core_width=self.core_width,
            blur=self.blur,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class ZigWaveguides(Sample):
    """A periodic array of waveguides that curve (wiggle) along the z-axis."""

    modulus: float = 0.08
    phase: Optional[float] = 0.0
    num_cores: float = 5.0
    core_width: float = 1.5
    wiggle_amp: float = 2.0

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        wiggle_offset = self.wiggle_amp * np.sin(2 * np.pi * z / (grid.z_prop / 2))
        eps = _build_waveguides(
            grid.x,
            self.modulus,
            self.phase,
            offset=wiggle_offset,
            num_cores=self.num_cores,
            core_width=self.core_width,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class StraightBalls(Sample):
    """A discrete sequence of highly localized spherical features arranged in straight vertical lines."""

    modulus: float = 0.08
    phase: Optional[float] = 0.0
    num_balls: int = 15
    ball_radius: float = 0.8
    x_period: float = 5.0

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        eps = _build_balls(
            grid.x,
            z,
            self.modulus,
            self.phase,
            grid.z_prop,
            self.num_balls,
            self.ball_radius,
            self.x_period,
            x_amplitude=0.0,
            is_zigzag=False,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class ZigBalls(Sample):
    """A discrete sequence of highly localized spherical features arranged in a sharp zig-zag pattern."""

    modulus: float = 0.08
    phase: Optional[float] = 0.0
    num_balls: int = 25
    ball_radius: float = 0.8
    x_period: float = 5.0
    x_amplitude: float = 3.0

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        eps = _build_balls(
            grid.x,
            z,
            self.modulus,
            self.phase,
            grid.z_prop,
            self.num_balls,
            self.ball_radius,
            self.x_period,
            self.x_amplitude,
            is_zigzag=True,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)
