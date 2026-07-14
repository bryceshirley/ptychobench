# src/ptychobench/operators.py
import numpy as np
from scipy.linalg import expm, sqrtm
from abc import ABC, abstractmethod


class ForwardOperator(ABC):
    """
    The Abstract Base Class (Interface) for all propagation operators.
    Enforces that every operator must have a 'step' method.
    """

    def __init__(self, grid):
        self.name = ""
        self.grid = grid
        self.I = np.eye(grid.N, dtype=complex)
        self.dz_factor = 1j * grid.k0 * grid.dz

    @abstractmethod
    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        """
        Constructs the propagation operator matrix for a given environment E.
        Must be implemented by all child classes.
        """
        pass

    def step(self, E: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Takes the environment matrix E and returns the propagation matrix P.
        Must be implemented by all child classes.
        """
        Q = self.construct_operator(E)
        return expm(self.dz_factor * Q).dot(psi)


class ExactOperator(ForwardOperator):
    """
    Computes the exact ground-truth operator.
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Exact"
        self.I_plus_M = self.I + grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        return sqrtm(self.I_plus_M + E) - self.I


class ParaxialOperator(ForwardOperator):
    """
    Q1 (Paraxial Approximation).
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Paraxial"
        self.half_M = 0.5 * grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        return 0.5 * E + self.half_M


class YevickThomsonOperator(ForwardOperator):
    """
    Q2a (Yevick/Thomson Split-Step Approximation).
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Yevick/Thomson"
        self.L_op = grid.get_angular_spectrum_operator()
        self.mu_op = grid.get_kinetic_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        N_op = np.sqrt(1 + E) - 1.0
        eps_vec = np.diag(E)
        Q2 = self.L_op + N_op
        cross_term = (1 / 8) * (
            (self.mu_op * eps_vec) + (self.mu_op * eps_vec[:, None])
        )
        return Q2 - cross_term


class FeitFleckOperator(ForwardOperator):
    """
    Q2 (Feit/Fleck Split-Step Approximation).
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Feit/Fleck"
        self.L_op = grid.get_angular_spectrum_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        eps = np.diag(E)
        N_op = np.diag(np.sqrt(1 + eps) - 1.0)

        return self.L_op + N_op


class LinDudaOperator(ForwardOperator):
    """
    Q3 (Lin/Duda Approximation).
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.name = "Lin/Duda"
        self.L_op = grid.get_angular_spectrum_operator()

    def construct_operator(self, E: np.ndarray) -> np.ndarray:
        eps = np.diag(E)

        # Keep 1D vector for optimized array broadcasting
        n_vec = np.sqrt(1 + eps) - 1.0
        N_op = np.diag(n_vec)

        Q2 = self.L_op + N_op

        # Fast O(N^2) cross-term calculation using broadcasting
        cross_term = 0.5 * ((self.L_op * n_vec) + (self.L_op * n_vec[:, None]))

        Q3 = Q2 - cross_term
        return Q3
