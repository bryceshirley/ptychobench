import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ==========================================
# Private Helper Functions (Pure Math)
# ==========================================
def _apply_boundary_mask(eps: np.ndarray, x_array: np.ndarray, L: float) -> np.ndarray:
    """Forces free space (epsilon perturbation = 0) in the padded boundary regions."""
    eps[np.abs(x_array) >= L / 2] = 0.0 + 0j
    return eps


def _build_waveguides(x_array, modulus, phase, offset, period, core_width):
    """Core logic for generating periodic waveguides using modulus and phase."""
    x_mod = (x_array - offset + period / 2) % period - period / 2
    profile = np.exp(-(x_mod**2) / (core_width**2))

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
# Caching for External Data Samples
# ==========================================
_SAMPLE_CACHE = {}


def _get_interpolated_sample(file_path, target_nx):
    """Loads and caches an external sample, zooming the X-axis to match the grid."""
    cache_key = (str(file_path), target_nx)

    if cache_key not in _SAMPLE_CACHE:
        data = np.load(file_path)
        current_nz, current_nx = data.shape

        if current_nx != target_nx:
            zoom_x = target_nx / current_nx
            data = zoom(data, (1.0, zoom_x), order=1)

        # Ignore "hot pixels"
        data_max = np.percentile(data, 99.9)

        if data_max > 0:
            data = data / data_max

        # Clip any of those extreme outlier pixels down to 1.0 so they don't break the scale
        data = np.clip(data, 0.0, 1.0)

        _SAMPLE_CACHE[cache_key] = data

    return _SAMPLE_CACHE[cache_key]


# ==========================================
# Public API (Object-Oriented Samples)
# ==========================================


class Sample(ABC):
    """Base class for all physical samples in the simulation."""

    @abstractmethod
    def get_permittivity(self, grid, z: float) -> np.ndarray:
        """Returns the 1D complex permittivity array for a specific z-slice."""
        pass


@dataclass
class Apoferritin(Sample):
    """An apoferritin biological sample interpolated from a saved .npy 2D cross-section."""

    modulus: float = 0.08
    phase: Optional[float] = None
    file_path: Optional[Path] = Path(__file__).parent / "apoferritin_100Nz565Nx.npy"

    def __post_init__(self):
        if self.phase is None:
            self.phase = self.modulus * 1e-2

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        target_nx = len(grid.x)
        full_2d_sample = _get_interpolated_sample(self.file_path, target_nx)
        current_nz = full_2d_sample.shape[0]

        z_fraction = np.clip(z / grid.z_prop, 0.0, 1.0)
        z_idx = int(z_fraction * (current_nz - 1))

        # The profile is now safely normalized between 0 and 1
        profile = full_2d_sample[z_idx, :]

        # Multiply the normalized profile by the user's chosen modulus
        # to scale the overall strength of the physical object.
        scaled_profile = profile * self.modulus

        # 1. Create a spatial phase map based on the normalized structure
        phase_map = profile * self.phase

        # 2. Combine the scaled amplitude with the complex phase shift
        eps = scaled_profile * np.exp(1j * phase_map)

        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class StraightWaveguides(Sample):
    """A periodic array of perfectly straight waveguides."""

    modulus: float = 0.08
    phase: Optional[float] = None
    period: float = 10.0
    core_width: float = 1.5

    def __post_init__(self):
        if self.phase is None:
            self.phase = self.modulus * 1e-2

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        eps = _build_waveguides(
            grid.x,
            self.modulus,
            self.phase,
            offset=0.0,
            period=self.period,
            core_width=self.core_width,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class ZigWaveguides(Sample):
    """A periodic array of waveguides that curve (wiggle) along the z-axis."""

    modulus: float = 0.08
    phase: Optional[float] = None
    period: float = 10.0
    core_width: float = 1.5
    wiggle_amp: float = 2.0

    def __post_init__(self):
        if self.phase is None:
            self.phase = self.modulus * 1e-2

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        wiggle_offset = self.wiggle_amp * np.sin(2 * np.pi * z / (grid.z_prop / 2))
        eps = _build_waveguides(
            grid.x,
            self.modulus,
            self.phase,
            offset=wiggle_offset,
            period=self.period,
            core_width=self.core_width,
        )
        return _apply_boundary_mask(eps, grid.x, grid.L)


@dataclass
class StraightBalls(Sample):
    """A discrete sequence of highly localized spherical features arranged in straight vertical lines."""

    modulus: float = 0.08
    phase: Optional[float] = None
    num_balls: int = 15
    ball_radius: float = 0.8
    x_period: float = 5.0

    def __post_init__(self):
        if self.phase is None:
            self.phase = self.modulus * 1e-2

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
    phase: Optional[float] = None
    num_balls: int = 25
    ball_radius: float = 0.8
    x_period: float = 5.0
    x_amplitude: float = 3.0

    def __post_init__(self):
        if self.phase is None:
            self.phase = self.modulus * 1e-2

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
