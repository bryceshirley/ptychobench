import pytest
import numpy as np
from scipy.linalg import sqrtm

# Adjust import based on your actual project structure
from ptychobench.operators import (
    GroundTruthOperator,
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
    YevickThomsonOperator,
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
        (GroundTruthOperator(grid), "Ground Truth"),
        (ParaxialOperator(grid), "Paraxial"),
        (FeitFleckOperator(grid), "Feit/Fleck"),
        (LinDudaOperator(grid), "Lin/Duda"),
    ]

    for op, expected_name in ops:
        assert op.name == expected_name
        assert op.grid == grid
        assert op.I.shape == (grid.N, grid.N)


def test_operator_step_output_shape(grid, dummy_E):
    """Test that the step method returns a matrix of the correct shape and type."""
    operators = [
        GroundTruthOperator(grid),
        ParaxialOperator(grid),
        FeitFleckOperator(grid),
        LinDudaOperator(grid),
    ]
    psi = np.eye(grid.N, dtype=complex)  # Identity wavefield for testing

    for op in operators:
        P = op.step(dummy_E, psi)
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
        GroundTruthOperator(grid),
        ParaxialOperator(grid),
        FeitFleckOperator(grid),
        LinDudaOperator(grid),
    ]

    I_expected = np.eye(grid.N, dtype=complex)
    psi = np.eye(grid.N, dtype=complex)  # Identity wavefield for testing
    for op in operators:
        # Re-initialize the operator so it picks up the zeroed lambda functions
        op.__init__(grid)
        P = op.step(E_zero, psi)

        # assert_allclose is great for floating point comparisons!
        np.testing.assert_allclose(
            P,
            I_expected,
            atol=1e-10,
            err_msg=f"{op.name} failed trivial propagation check",
        )


@pytest.mark.parametrize(
    "OpClass, integrator",
    [
        (GroundTruthOperator, "krylov"),
        (ParaxialOperator, "krylov"),
        (ParaxialOperator, "split-step"),
        (FeitFleckOperator, "krylov"),
        (FeitFleckOperator, "split-step"),
        (YevickThomsonOperator, "krylov"),
        (LinDudaOperator, "krylov"),
    ],
    ids=lambda p: p if isinstance(p, str) else p.__name__,
)
def test_a_matrix_free_run_allocates_no_dense_matrix(OpClass, integrator, grid):
    """The dense N x N scratch matrices are only ever read by the dense path.

    Building them in __init__ meant a Krylov or split-step run paid for
    matrices it never touched -- 1 GB and 33 seconds *per operator* at
    N = 8192, which defeats the entire point of being matrix-free. Every one
    of them is a cached_property now, so nothing is allocated until something
    actually asks for it.

    Checked over every operator rather than the two that had it first: each
    class names its own matrices, so this is exactly the kind of thing that
    gets fixed in one place and reintroduced in the next.
    """
    operator = OpClass(grid, integrator=integrator)

    dense = {
        name: value
        for name, value in vars(operator).items()
        if isinstance(value, np.ndarray) and value.ndim == 2
    }
    assert dense == {}, f"{OpClass.__name__} eagerly built {sorted(dense)}"


def test_the_dense_matrices_are_still_built_once_for_the_dense_path(grid):
    """Lazy, not absent -- and cached, so the dense path pays only on first use."""
    exact = GroundTruthOperator(grid)

    assert exact.I.shape == (grid.N, grid.N)
    assert exact.I is exact.I
    assert exact.I_plus_M is exact.I_plus_M
