import numpy as np
from scipy.ndimage import zoom
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from matplotlib import pyplot as plt

from ase import Atoms
import abtem


# ==========================================
# Private Helper Functions (Pure Math)
# ==========================================
def _apply_boundary_mask(eps: np.ndarray, grid) -> np.ndarray:
    """
    Free space in the padded boundary regions for 2D transverse grids.
    Returns a complex copy so downstream propagation code has consistent dtype.
    """
    eps = np.asarray(eps, dtype=np.complex128).copy()

    # Broadcast masks for X and Y
    mask_x = np.abs(grid.x[:, None]) >= grid.Lx / 2
    mask_y = np.abs(grid.y[None, :]) >= grid.Ly / 2

    # Apply mask where either condition is met
    eps[mask_x | mask_y] = 0.0 + 0.0j

    return eps


class Sample(ABC):
    """Base class for all 3D physical samples in the simulation."""

    @abstractmethod
    def get_permittivity(self, grid, z: float) -> np.ndarray:
        """Returns the 2D (Nx, Ny) complex permittivity array for a specific z-slice."""
        pass

    def plot_cross_section_modulus(self, grid, z: float = 0.0, fourier: bool = False):
        """Plots the 2D modulus of the permittivity at a specific z-slice."""
        eps = self.get_permittivity(grid, z)
        extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]

        if fourier:
            eps = np.fft.fftn(eps)
            eps = np.fft.fftshift(eps)
            title = f"Modulus of Permittivity at z={z:.2f} (Fourier Domain)"
            extent = [grid.kx.min(), grid.kx.max(), grid.ky.min(), grid.ky.max()]
        else:
            title = f"Modulus of Permittivity at z={z:.2f} (Spatial Domain)"

        plt.imshow(np.abs(eps).T, extent=extent, origin="lower", cmap="viridis")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.colorbar(label="|ε(x, y, z)|")
        plt.show()
        plt.close()

    def plot_cross_section_phase(self, grid, z: float = 0.0):
        """Plots the 2D phase of the permittivity at a specific z-slice."""
        eps = self.get_permittivity(grid, z)
        extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]

        plt.imshow(np.angle(eps).T, extent=extent, origin="lower", cmap="plasma")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Phase of Permittivity at z={z:.2f}")
        plt.colorbar(label="∠ε(x, y, z)")
        plt.show()
        plt.close()

    def plot(self, grid):
        """
        Compute and plot the X-Z longitudinal cross-section through the center (Y=0)
        of the 3D sample.
        """
        Nz = grid.Nz
        Nx = grid.Nx
        y_mid = grid.Ny // 2

        # Build sample slice on-the-fly
        sample_history = np.zeros((Nz, Nx), dtype=np.complex128)

        for i, z in enumerate(grid.z_steps):
            eps_2d = self.get_permittivity(grid, z)
            sample_history[i, :] = eps_2d[:, y_mid]

        extent = [
            grid.x[0],
            grid.x[-1],
            grid.z_steps[0],
            grid.z_steps[-1],
        ]

        fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        im_real = ax_real.imshow(
            np.abs(sample_history),
            extent=extent,
            aspect="auto",
            cmap="viridis",
            origin="upper",
        )
        ax_real.set_title("Modulus (Center Y-Slice)")
        ax_real.set_xlabel("Transverse coordinate x")
        ax_real.set_ylabel("Propagation distance z")
        fig.colorbar(im_real, ax=ax_real)

        im_imag = ax_imag.imshow(
            np.unwrap(np.angle(sample_history), axis=1),
            extent=extent,
            aspect="auto",
            cmap="plasma",
            origin="upper",
        )
        ax_imag.set_title("Phase (radians)")
        ax_imag.set_xlabel("Transverse coordinate x")
        fig.colorbar(im_imag, ax=ax_imag)

        fig.suptitle("Refractive Index Environment (Longitudinal X-Z Cross Section)")
        plt.tight_layout()
        plt.show()


@dataclass
class AbtemSample(Sample):
    """
    A direct adapter for pre-built abtem/ASE atomic models.
    """

    atoms: Atoms
    modulus: float = 0.05
    phase: float = 0.0

    _sample_cache: ClassVar[dict] = {}

    def _cache_key(self, grid):
        return (
            self.atoms.get_chemical_formula(),
            int(grid.Nz),
            int(grid.Nx),
            int(grid.Ny),
            self.modulus,
        )

    def cache_sample(self, grid) -> np.ndarray:
        key = self._cache_key(grid)
        if key in self._sample_cache:
            return self._sample_cache[key]

        # 1. Calculate Electrostatic Potential using abtem (uses Angstroms)
        # We ensure the slice thickness perfectly matches the grid's Z-steps
        Lz_A = grid.z_prop * 10.0
        dz_A = Lz_A / grid.Nz

        potential = abtem.Potential(
            self.atoms,
            sampling=(grid.dx * 10.0, grid.dy * 10.0),
            slice_thickness=dz_A,
            projection="infinite",
            parametrization="lobato",
        )

        # Build the array (Shape: Nz, Nx_abtem, Ny_abtem)
        built_potential = potential.build()
        pot_array = built_potential.array

        # 2. Guarantee exact shape matching for the Krylov Solvers
        target_shape = (grid.Nz, grid.Nx, grid.Ny)
        if pot_array.shape != target_shape:
            zoom_z = target_shape[0] / pot_array.shape[0]
            zoom_x = target_shape[1] / pot_array.shape[1]
            zoom_y = target_shape[2] / pot_array.shape[2]
            pot_array = zoom(pot_array, (zoom_z, zoom_x, zoom_y), order=1)

        # 3. Normalize and apply the user's modulus/phase convention
        pot_max = np.max(pot_array)
        if pot_max > 0:
            pot_array = pot_array / pot_max

        complex_pert = self.modulus * np.exp(1j * self.phase * pot_array) * pot_array

        self._sample_cache[key] = complex_pert.astype(np.complex128)
        return self._sample_cache[key]

    def get_permittivity(self, grid, z: float) -> np.ndarray:
        sample = self.cache_sample(grid)

        z_fraction = np.clip(z / grid.z_prop, 0.0, 1.0)
        z_idx = np.clip(
            int(round(z_fraction * (sample.shape[0] - 1))), 0, sample.shape[0] - 1
        )

        # We do NOT apply a boundary mask here because atomic models are inherently periodic
        return sample[z_idx, :, :]
