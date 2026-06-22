import pytest
import numpy as np
from scipy.linalg import sqrtm

# Adjust import based on your actual project structure
from ptychobench.operators import (
    ExactOperator,
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)

# --- MOCK OBJECTS FOR ISOLATED TESTING ---


class MockOperatorGrid:
    """
    A mock grid providing the necessary attributes and methods
    required by the different ForwardOperators.
    """

    def __init__(self):
        self.N = 4
        self.k0 = 2.0 * np.pi
        self.dz = 0.1

    def get_kinetic_operator(self):
        # A simple symmetric diagonal matrix representing M
        return np.diag([0.1, 0.2, 0.2, 0.1])

    def get_angular_spectrum_operator(self):
        # L = sqrt(I + M) - I
        identity = np.eye(self.N)
        M = self.get_kinetic_operator()
        return sqrtm(identity + M) - identity


# --- FIXTURES ---


@pytest.fixture
def grid():
    """Provides a fresh MockOperatorGrid for each test."""
    return MockOperatorGrid()


@pytest.fixture
def dummy_E():
    """Provides a simple environment perturbation matrix (E)."""
    return np.diag([0.05, 0.01, -0.02, 0.0])


# --- TESTS ---


def test_operator_initialization(grid):
    """Test that all operators initialize correctly with expected names."""
    ops = [
        (ExactOperator(grid), "Exact"),
        (ParaxialOperator(grid), "Paraxial (Q1)"),
        (FeitFleckOperator(grid), "Feit/Fleck (Q2)"),
        (LinDudaOperator(grid), "Lin/Duda (Q3)"),
    ]

    for op, expected_name in ops:
        assert op.name == expected_name
        assert op.grid == grid
        assert op.I.shape == (grid.N, grid.N)


def test_operator_step_output_shape(grid, dummy_E):
    """Test that the step method returns a matrix of the correct shape and type."""
    operators = [
        ExactOperator(grid),
        ParaxialOperator(grid),
        FeitFleckOperator(grid),
        LinDudaOperator(grid),
    ]

    for op in operators:
        P = op.step(dummy_E)
        assert P.shape == (grid.N, grid.N), f"{op.name} returned incorrect shape"
        assert P.dtype == complex, f"{op.name} should return a complex array"


def test_trivial_propagation(grid):
    """
    MATH CHECK: If the environment perturbation (E) is zero, and the
    kinetic operator (M) is zero, the propagation matrix P should
    simply be the Identity matrix (meaning the wave doesn't change).
    """
    # Override the grid methods to return pure zeros for this specific test
    grid.get_kinetic_operator = lambda: np.zeros((grid.N, grid.N))
    grid.get_angular_spectrum_operator = lambda: np.zeros((grid.N, grid.N))

    E_zero = np.zeros((grid.N, grid.N))

    operators = [
        ExactOperator(grid),
        ParaxialOperator(grid),
        FeitFleckOperator(grid),
        LinDudaOperator(grid),
    ]

    I_expected = np.eye(grid.N, dtype=complex)

    for op in operators:
        # Re-initialize the operator so it picks up the zeroed lambda functions
        op.__init__(grid)
        P = op.step(E_zero)

        # assert_allclose is great for floating point comparisons!
        np.testing.assert_allclose(
            P,
            I_expected,
            atol=1e-10,
            err_msg=f"{op.name} failed trivial propagation check",
        )
