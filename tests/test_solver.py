import numpy as np

# Adjust imports based on your actual project structure
from ptychobench.solver import run_benchmark, BenchmarkResult

# --- MOCK OBJECTS FOR ISOLATED TESTING ---


class MockSimulationGrid:
    """A lightweight fake grid to use in testing."""

    def __init__(self):
        self.Nz = 2  # Just 2 z-steps for a quick test
        self.N = 3  # 3 spatial points
        self.z_steps = [0.0, 1.0]

    def get_initial_field(self):
        # Start with a simple array of 1s
        return np.ones(self.N, dtype=complex)


class MockExactOperator:
    """A fake 'Exact' operator that does nothing to the wave."""

    name = "Exact"

    def __init__(self, grid):
        self.grid = grid

    def step(self, E):
        # Identity matrix: no change during propagation
        return np.eye(self.grid.N, dtype=complex)


class MockApproximateOperator:
    """A fake approximation that slightly reduces the wave amplitude."""

    name = "Approx"

    def __init__(self, grid):
        self.grid = grid

    def step(self, E):
        # Multiplies by 0.9 to guarantee it differs from Exact
        return np.eye(self.grid.N, dtype=complex) * 0.9


def dummy_sample_function(grid, z, **kwargs):
    """A fake sample that returns 0 perturbation."""
    return np.zeros(grid.N, dtype=complex)


# --- TESTS ---


def test_run_benchmark_basic_execution():
    """
    Test that the solver runs without crashing and returns the expected
    BenchmarkResult structure when no error calculation is triggered.
    """
    grid = MockSimulationGrid()
    operators = [MockExactOperator]

    result = run_benchmark(grid, dummy_sample_function, operators)

    assert isinstance(result, BenchmarkResult)
    assert "Exact" in result.histories
    assert result.histories["Exact"].shape == (grid.Nz, grid.N)
    assert result.errors["Exact"] == 0.0


def test_run_benchmark_calculates_rmse():
    """
    This test checks if the RMSE is correctly calculated for the 'Approx' operator.
    """
    grid = MockSimulationGrid()
    operators = [MockExactOperator, MockApproximateOperator]

    # This will run the solver and call calculate_rmse
    result = run_benchmark(grid, dummy_sample_function, operators)

    assert "Approx" in result.errors

    # The following assertion will fail because calculate_rmse currently
    # doesn't return a valid float.
    calculated_error = result.errors["Approx"]

    assert isinstance(calculated_error, float), "Error should be a float number"
    assert calculated_error > 0.0, "The Approx operator should have an error > 0"
