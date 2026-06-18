from dataclasses import dataclass
import numpy as np
from scipy.linalg import dft


@dataclass
class SimulationGrid:
    """
    Holds all physical and computational parameters for the 2D Ptychography simulation.
    """

    # ---------------------------------------------------------
    # Core Parameters (No defaults must come first)
    # ---------------------------------------------------------
    # The propagation angle (in degrees) at the edge of the probe width.
    divergence_angle: float

    # ---------------------------------------------------------
    # Default Parameters
    # ---------------------------------------------------------
    lam: float = 1.0  # Wavelength of the light (arbitrary units)
    L: float = 100.0  # Total transverse width of the simulation domain
    z_prop: float = 50.0  # Total propagation distance
    N: int = 128  # Number of transverse pixels
    Nz: int = 100  # Number of longitudinal steps (z-steps)
    probe_width: float = 5.0  # Width of the initial Gaussian beam

    # =========================================================
    # Initialization Hooks (The Optimization)
    # =========================================================
    def __post_init__(self):
        """
        Runs automatically after the dataclass is initialized.
        Pre-calculates heavy, constant matrices to avoid redundant computation.
        """
        # Calculate Scipy's DFT matrix for exact matrix math rather than FFTs
        self.F = dft(self.N) / np.sqrt(self.N)
        self.F_inv = self.F.conj().T

    # =========================================================
    # Derived Properties (Calculated automatically)
    # =========================================================
    @property
    def k0(self) -> float:
        """Wavenumber."""
        return 2 * np.pi / self.lam

    @property
    def dx(self) -> float:
        """Transverse pixel size."""
        return self.L / self.N

    @property
    def dz(self) -> float:
        """Longitudinal step size."""
        return self.z_prop / self.Nz

    @property
    def x(self) -> np.ndarray:
        """Transverse spatial coordinates."""
        return np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)

    @property
    def kx(self) -> np.ndarray:
        """Transverse spectral (Fourier) coordinates."""
        return 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

    @property
    def z_steps(self) -> np.ndarray:
        """Array of z-coordinates for the propagation loop."""
        return np.linspace(0, self.z_prop, self.Nz, endpoint=False)

    # =========================================================
    # Helper Methods (Physics Generation)
    # =========================================================
    def get_initial_field(self) -> np.ndarray:
        """
        Generates the initial wave function (psi_0) at z = 0.
        Combines a Gaussian amplitude with a quadratic phase tilt based on the divergence angle.
        """
        amplitude = np.exp(-(self.x**2) / (2 * self.probe_width**2))

        # Convert degrees to radians for internal math
        div_angle_rads = np.radians(self.divergence_angle)

        # Calculate phase multiplier to steer the beam
        phase_multiplier = div_angle_rads / (2 * self.probe_width)
        phase = np.exp(1j * self.k0 * (self.x**2) * phase_multiplier)

        return amplitude * phase

    def get_kinetic_operator(self) -> np.ndarray:
        """
        Calculates the constant kinetic operator matrix (M).
        M = F_inv @ D_kx @ F
        """
        # Diagonal matrix of spectral frequencies
        D_kx = np.diag(-(self.kx**2) / (self.k0**2) + 0j)

        # Reuse the pre-calculated DFT matrices
        return self.F_inv @ D_kx @ self.F

    def get_angular_spectrum_operator(self) -> np.ndarray:
        """
        Calculates the angular spectrum operator (L_op).
        L_op = sqrt(I - (kx^2 / k0^2)) - I
        """
        # Angular spectrum diagonal matrix
        D_L = np.diag(np.sqrt(1 - (self.kx**2) / (self.k0**2) + 0j) - 1.0)

        # Reuse the pre-calculated DFT matrices
        return self.F_inv @ D_L @ self.F
