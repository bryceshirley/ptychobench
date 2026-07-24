# src/ptychobench_3d/grid.py
from dataclasses import dataclass
from abtem import Probe
import numpy as np


@dataclass
class SimulationGrid:
    """
    Holds all physical and computational parameters for 3D wave propagation
    (2D Transverse + 1D Longitudinal).
    """

    # Default Parameters
    probe_field: np.ndarray  # Initial probe field (2D array)
    probe_abtem: Probe  # Abtem probe object
    L: float  # Total transverse width
    z_prop: float  # Total propagation distance
    N: int  # Number of transverse grid points
    Nz: int  # Number of longitudinal steps (z-steps)

    @property
    def k0(self) -> float:
        """Wavenumber."""
        return 2 * np.pi / self.wavelength

    @property
    def wavelength(self) -> float:
        """Wavelength"""
        return self.probe_abtem.wavelength

    @property
    def energy(self) -> float:
        """Relativistic beam energy in eV."""
        return self.probe_abtem.energy

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def dy(self) -> float:
        return self.dx

    @property
    def dz(self) -> float:
        return self.z_prop / self.Nz

    # --- 1D Coordinate Vectors ---
    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)

    @property
    def kx(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.probe_field.shape[1], d=self.dx)

    @property
    def ky(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.probe_field.shape[0], d=self.dy)

    @property
    def z_steps(self) -> np.ndarray:
        return np.linspace(0, self.z_prop, self.Nz, endpoint=False)

    # --- 2D Broadcasted Grids (Memory Efficient) ---
    @property
    def R_sq(self) -> np.ndarray:
        """Squared transverse spatial radius (x^2 + y^2)."""
        return self.x[:, np.newaxis] ** 2 + self.y[np.newaxis, :] ** 2

    @property
    def K_sq(self) -> np.ndarray:
        """Squared transverse wave vector (kx^2 + ky^2)."""
        return self.kx[:, np.newaxis] ** 2 + self.ky[np.newaxis, :] ** 2

    def get_initial_field(self) -> np.ndarray:
        """Returns the initial probe field (2D array)."""
        return self.probe_field.copy()


def prepare_potential(grid, potential_array):
    """
    Converts abtem's projected potential into the purely real interaction term E
    used by the matrix-free Krylov operators.
    """
    # 0. Ensure potential_array is a concrete numpy array
    if hasattr(potential_array, "compute"):
        potential_array = potential_array.compute()
    potential_array = np.asarray(potential_array)

    # 1. Calculate the exact relativistic sigma
    m0c2 = 510998.95
    wavelength = grid.probe_abtem.wavelength
    energy = grid.probe_abtem.energy
    sigma = (2.0 * np.pi / (wavelength * energy)) * (
        (energy + m0c2) / (energy + 2.0 * m0c2)
    )

    # 2. Convert V to E using the derived formula
    # E = (2 * sigma * V) / (k0 * dz)
    E_array = (2.0 * sigma * potential_array) / (grid.k0 * grid.dz)

    # Return exactly as a real numpy array
    return E_array
