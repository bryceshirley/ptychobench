# src/ptychobench/operators.py
import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod
from typing import Callable
import torch
from abtem.antialias import antialias_aperture

# Setup PyTorch device to use Apple MPS (Metal Performance Shaders) if available
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================================================
# 1. PyTorch-Accelerated Krylov Solver Core (MPS / CPU)
# =========================================================


def mps_arnoldi_step(
    operator_func: Callable,
    v: torch.Tensor,
    iterations: int,
    func: Callable,
    tol: float = 1e-6,
) -> tuple:
    """Gram-Schmidt with Re-orthogonalization using PyTorch/MPS.

    Parameters:
    - operator_func: A callable that applies the matrix-free operator to a vector.
    - v: The input vector (torch.Tensor) to project onto the Krylov subspace
    - iterations: The number of Arnoldi iterations to perform
    - func: The specific matrix function applied to the Hessenberg matrix
    - tol: The tolerance for convergence
    Returns:
    - V: Orthonormal basis of the Krylov subspace (torch.Tensor)
    - H: Upper Hessenberg matrix representing the operator in the Krylov basis (torch.Tensor)
    - active_iterations: The number of iterations actually performed (int)
    - v_norm: The norm of the input vector (float)
    """
    shape = v.shape
    N = v.numel()
    v_flat = v.reshape(-1)

    # Pre-allocate tensors on the target device
    V = torch.zeros((iterations + 1, N), dtype=v.dtype, device=v.device)
    H = torch.zeros((iterations, iterations), dtype=v.dtype, device=v.device)

    v_norm = torch.linalg.norm(v_flat).item()
    if v_norm == 0:
        return V, H, 0, v_norm

    V[0] = v_flat / v_norm
    active_iterations = iterations

    for j in range(iterations):
        # 1. Apply Matrix-Free Operator
        w = operator_func(V[j].reshape(shape)).reshape(-1)

        # 2. Vectorized CGS2 using PyTorch matrix operations
        h1 = torch.matmul(V[: j + 1].conj(), w)
        w = w - torch.matmul(h1, V[: j + 1])

        h2 = torch.matmul(V[: j + 1].conj(), w)
        w = w - torch.matmul(h2, V[: j + 1])

        H[: j + 1, j] = h1 + h2
        w_norm = torch.linalg.norm(w).item()

        # 3. Store subdiagonal and build orthonormal basis for fixed subspace size
        if j < iterations - 1:
            H[j + 1, j] = w_norm
            V[j + 1] = w / w_norm

        # Check convergence every 10 iterations to save CPU overhead
        if j > 10 and j % 10 == 0:
            # Copy the current active small H matrix to CPU
            H_curr = H[: j + 1, : j + 1].detach().cpu().numpy()

            try:
                # Compute the EXACT matrix function used by the specific operator
                fH = func(H_curr)

                # The error is ||w|| * abs(bottom-left element of fH)
                bottom_left_val = np.abs(fH[j, 0])
                current_error = w_norm * bottom_left_val

                if current_error < tol:
                    active_iterations = j + 1
                    break
            except Exception:
                # Fallback in case sqrtm fails on a highly ill-conditioned early subspace
                pass

    return V, H, active_iterations, v_norm


def mps_krylov_funm(
    operator_func: Callable,
    v: torch.Tensor,
    func: Callable,
    iterations: int = 300,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Projects onto Krylov subspace, applies dense matrix function via SciPy, and projects back."""

    # Pass 'func' down into the Arnoldi step
    V, H, active_iterations, v_norm = mps_arnoldi_step(
        operator_func, v, iterations, func, tol
    )

    if active_iterations == 0:
        return v

    # Bring active small matrix back to CPU for dense matrix functions (SciPy/NumPy handles small dense matrices efficiently)
    H_active = H[:active_iterations, :active_iterations].detach().cpu().numpy()
    fH_np = func(H_active)

    fH = torch.from_numpy(
        fH_np.astype(np.complex64 if v.dtype == torch.complex64 else np.complex128)
    ).to(v.device)

    e1 = torch.zeros(active_iterations, dtype=v.dtype, device=v.device)
    e1[0] = v_norm
    fH_e1 = torch.matmul(fH, e1)

    result = torch.matmul(fH_e1, V[:active_iterations]).reshape(v.shape)
    return result


# =========================================================
# 2. Stateful Operator Actions (PyTorch Tensor Operations)
# =========================================================


class ExactAction:
    def __init__(self, M_diag):
        self.M_diag = M_diag
        self.E = None

    def update(self, E: torch.Tensor):
        self.E = E

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        psi_k = torch.fft.fftn(psi)
        k_term = torch.fft.ifftn(self.M_diag * psi_k)
        return psi + k_term + self.E * psi


class ParaxialAction:
    def __init__(self, M_diag):
        self.half_M = 0.5 * M_diag
        self.half_E = None

    def update(self, E: torch.Tensor):
        self.half_E = 0.5 * E

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        psi_k = torch.fft.fftn(psi)
        return self.half_E * psi + torch.fft.ifftn(self.half_M * psi_k)


class FeitFleckAction:
    def __init__(self, L_diag):
        self.L_diag = L_diag
        self.N_diag = None

    def update(self, E: torch.Tensor):
        self.N_diag = torch.sqrt(1.0 + E) - 1.0

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        psi_k = torch.fft.fftn(psi)
        return torch.fft.ifftn(self.L_diag * psi_k) + self.N_diag * psi


class YevickThomsonAction:
    def __init__(self, L_diag, M_diag):
        self.L_diag = L_diag
        self.M_diag = M_diag
        self.E = None
        self.N_diag = None

    def update(self, E: torch.Tensor):
        self.E = E
        self.N_diag = torch.sqrt(1.0 + E) - 1.0

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        psi_k = torch.fft.fftn(psi)
        L_psi = torch.fft.ifftn(self.L_diag * psi_k)
        N_psi = self.N_diag * psi

        E_psi = self.E * psi
        M_E_psi = torch.fft.ifftn(self.M_diag * torch.fft.fftn(E_psi))
        M_psi = torch.fft.ifftn(self.M_diag * psi_k)
        E_M_psi = self.E * M_psi

        return L_psi + N_psi - 0.125 * (M_E_psi + E_M_psi)


class LinDudaAction:
    def __init__(self, L_diag):
        self.L_diag = L_diag
        self.N_diag = None

    def update(self, E: torch.Tensor):
        self.N_diag = torch.sqrt(1.0 + E) - 1.0

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        psi_k = torch.fft.fftn(psi)
        L_psi = torch.fft.ifftn(self.L_diag * psi_k)
        N_psi = self.N_diag * psi

        L_N_psi = torch.fft.ifftn(self.L_diag * torch.fft.fftn(N_psi))
        N_L_psi = self.N_diag * L_psi

        return L_psi + N_psi - 0.5 * (L_N_psi + N_L_psi)


# =========================================================
# 3. Base Class & Operator Interfaces
# =========================================================


class ForwardOperator(ABC):
    """Matrix-free Base Class accelerated using PyTorch MPS/GPU."""

    default_iterations: int = 300

    def __init__(self, grid):
        self.name = ""
        self.grid = grid

        self.k0 = float(grid.k0)
        self.dz_factor_val = 1j * self.k0 * float(grid.dz)

        # Initialization Sequence
        self._init_diagonals()
        self.action_module = self._create_action_module()

        # Default matrix exponential function used by most operators
        self.matrix_func = self._standard_expm

    def _init_diagonals(self):
        """Initializes the 2D Transverse Spectral Arrays and Antialias Mask."""
        K_sq = torch.from_numpy(self.grid.K_sq.astype(np.float32)).to(DEVICE)

        self.propagating_mask = K_sq <= (self.k0**2)
        kz_sq_over_k0_sq = 1.0 - (K_sq) / (self.k0**2)
        kz_sq_over_k0_sq = torch.where(
            self.propagating_mask,
            kz_sq_over_k0_sq,
            torch.tensor(0.0, dtype=kz_sq_over_k0_sq.dtype, device=DEVICE),
        )

        self.M_diag = (kz_sq_over_k0_sq - 1.0).to(torch.complex64)
        self.L_diag = torch.sqrt(kz_sq_over_k0_sq.to(torch.complex64)) - 1.0

        # --- NEW: Shared Antialias Mask ---
        gpts = (self.grid.N, self.grid.N)
        sampling = (self.grid.dx, self.grid.dx)
        aperture_np = antialias_aperture(gpts, sampling, np)
        self.antialias_mask = torch.from_numpy(aperture_np.astype(np.float32)).to(
            DEVICE
        )

    def _standard_expm(self, H: np.ndarray) -> np.ndarray:
        return scipy.linalg.expm(self.dz_factor_val * H)

    def project_propagating(self, psi: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(self.propagating_mask * torch.fft.fftn(psi))

    @abstractmethod
    def _create_action_module(self):
        pass

    def step(self, E, psi) -> np.ndarray:
        """Executes a single split-step using the Matrix-Free Krylov solver."""
        iterations = self.default_iterations

        if hasattr(E, "compute"):
            E = E.compute()
        E = np.asarray(E)

        if hasattr(psi, "compute"):
            psi = psi.compute()
        psi = np.asarray(psi)

        E_tensor = torch.from_numpy(
            E.astype(np.complex64 if np.iscomplexobj(E) else np.float32)
        ).to(DEVICE)
        psi_tensor = torch.from_numpy(psi.astype(np.complex64)).to(DEVICE)

        self.action_module.update(E_tensor)

        # --- NEW: Bandlimited Action Wrapper ---
        # Forces the Arnoldi vectors to remain strictly inside the antialias cutoff,
        # preventing the Krylov basis from expanding into aliased frequencies.
        def bandlimited_action(v: torch.Tensor) -> torch.Tensor:
            out = self.action_module(v)
            out_k = torch.fft.fftn(out)
            return torch.fft.ifftn(out_k * self.antialias_mask)

        res_tensor = mps_krylov_funm(
            bandlimited_action, psi_tensor, func=self.matrix_func, iterations=iterations
        )

        return res_tensor.detach().cpu().numpy()


# =========================================================
# 4. Operator Implementations
# =========================================================


class ExactOperatorMF(ForwardOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Exact_mf"
        self.matrix_func = self._exact_expm

    def _create_action_module(self):
        return ExactAction(self.M_diag)

    def _exact_expm(self, H: np.ndarray) -> np.ndarray:
        """Exact matrix exponential using eigen-decomposition for small Krylov projection matrices."""
        H = np.asarray(H, dtype=np.complex128)
        H_sqrt = scipy.linalg.sqrtm(H)
        result = scipy.linalg.expm(self.dz_factor_val * (H_sqrt - np.eye(H.shape[0])))
        return result.astype(np.complex64, copy=False)


class ParaxialOperatorMF(ForwardOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Paraxial_mf"

    def _create_action_module(self):
        return ParaxialAction(self.M_diag)


class YevickThomsonOperatorMF(ForwardOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Yevick/Thomson_mf"

    def _create_action_module(self):
        return YevickThomsonAction(self.L_diag, self.M_diag)


class FeitFleckOperatorMF(ForwardOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Feit/Fleck_mf"

    def _create_action_module(self):
        return FeitFleckAction(self.L_diag)


class LinDudaOperatorMF(ForwardOperator):
    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Lin/Duda_mf"

    def _create_action_module(self):
        return LinDudaAction(self.L_diag)


class StandardMultisliceOperator(ForwardOperator):
    """
    Standard multislice operator modified to accept the Krylov 'E' term
    while retaining abtem's exact transmission and propagation sequence.
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Standard_Multislice"
        self.propagator = torch.exp(self.dz_factor_val * self.M_diag)

    def _create_action_module(self):
        return None

    def _init_diagonals(self):
        # Call base class to safely generate L_diag and the antialias_mask
        super()._init_diagonals()

        # OVERRIDE M_diag to include the Paraxial (Fresnel) / 2.0 approximation
        kz_sq_over_k0_sq = 1.0 - (
            torch.from_numpy(self.grid.K_sq.astype(np.float32)).to(DEVICE)
            / (self.k0**2)
        )
        kz_sq_over_k0_sq = torch.where(
            self.propagating_mask,
            kz_sq_over_k0_sq,
            torch.tensor(0.0, dtype=kz_sq_over_k0_sq.dtype, device=DEVICE),
        )
        self.M_diag = ((kz_sq_over_k0_sq - 1.0) / 2.0).to(torch.complex64)

    def step(self, E, psi) -> np.ndarray:
        if hasattr(E, "compute"):
            E = E.compute()
        E = np.asarray(E)

        if hasattr(psi, "compute"):
            psi = psi.compute()
        psi = np.asarray(psi)

        E_tensor = torch.from_numpy(
            E.astype(np.complex64 if np.iscomplexobj(E) else np.float32)
        ).to(DEVICE)
        psi_tensor = torch.from_numpy(psi.astype(np.complex64)).to(DEVICE)

        # --- Step A: Transmission Function ---
        transmission = torch.exp(self.dz_factor_val * E_tensor / 2.0)

        T_k = torch.fft.fftn(transmission)
        transmission_bandlimited = torch.fft.ifftn(T_k * self.antialias_mask)
        psi_transmitted = psi_tensor * transmission_bandlimited

        # --- Step B & C: Bandlimiting & Fresnel Propagation ---
        psi_k = torch.fft.fftn(psi_transmitted)
        psi_k_propagated = psi_k * self.antialias_mask * self.propagator

        psi_propagated = torch.fft.ifftn(psi_k_propagated)
        return psi_propagated.detach().cpu().numpy()
