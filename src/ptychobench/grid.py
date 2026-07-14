from dataclasses import dataclass
import numpy as np
from scipy.linalg import dft


@dataclass
class SimulationGrid:
    """
    Holds all physical and computational parameters for the 2D Ptychography simulation.
    """

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

        # Calculate a quadaratic phase shift
        phase_shift = np.exp(1j * div_angle_rads * (self.x**2) / (2 * self.probe_width))

        return amplitude * phase_shift

    @property
    def propagating_mask(self) -> np.ndarray:
        """
        True for propagating Fourier modes satisfying |kx| <= k0.
        False for evanescent modes.
        """
        return np.abs(self.kx) <= self.k0

    def project_propagating(self, psi: np.ndarray) -> np.ndarray:
        """
        Remove evanescent Fourier components from a field.
        """
        psi_k = self.F @ psi
        psi_k = self.propagating_mask * psi_k
        return self.F_inv @ psi_k

    def get_kinetic_operator(self) -> np.ndarray:
        """
        Calculates the band-limited kinetic operator matrix mu.

        Original:
            mu = F_inv @ diag(-kx^2/k0^2) @ F

        With evanescent removal:
            1 + mu = max(1 - kx^2/k0^2, 0)

        Therefore:
            mu = max(1 - kx^2/k0^2, 0) - 1
        """

        kz_sq_over_k0_sq = 1.0 - (self.kx**2) / (self.k0**2)

        kz_sq_over_k0_sq = np.where(self.propagating_mask, kz_sq_over_k0_sq, 0.0)

        mu_diag = kz_sq_over_k0_sq - 1.0

        D_mu = np.diag(mu_diag.astype(complex))

        return self.F_inv @ D_mu @ self.F

    def get_angular_spectrum_operator(self) -> np.ndarray:
        """
        Calculates the band-limited angular spectrum envelope operator L.

        L = sqrt(1 + mu) - 1

        In Fourier space:
            L(kx) = sqrt(1 - kx^2/k0^2) - 1

        Evanescent modes are clipped by setting kz = 0.
        """

        kz_sq_over_k0_sq = 1.0 - (self.kx**2) / (self.k0**2)

        kz_sq_over_k0_sq = np.where(self.propagating_mask, kz_sq_over_k0_sq, 0.0)

        L_diag = np.sqrt(kz_sq_over_k0_sq) - 1.0

        D_L = np.diag(L_diag.astype(complex))

        return self.F_inv @ D_L @ self.F
