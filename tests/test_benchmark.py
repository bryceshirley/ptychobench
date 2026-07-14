import numpy as np

# Adjust imports based on your actual project structure
from ptychobench.benchmark import run_benchmark
from ptychobench.results import BenchmarkResult
from ptychobench.grid import SimulationGrid
from ptychobench.samples import Sample


class DummySample(Sample):
    """A simple sample that does nothing, just to satisfy the Sample class interface."""

    def __init__(self):
        self.modulus = 1.0
        self.dummy_param = "test"

    def get_permittivity(self, grid, z):
        return np.zeros(grid.N, dtype=complex)


class DummyApproxOperator:
    """A basic operator that slightly reduces the wave to test the error calculator."""

    name = "DummyApprox (Test)"

    def __init__(self, grid):
        self.grid = grid

    def step(self, E, psi):
        return psi


# --- TESTS ---


def test_run_benchmark_basic_execution():
    """
    Test that the solver runs without crashing and correctly injects
    and calculates the ExactOperator baseline automatically.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=4, Nz=2)
    sample = DummySample()

    # Pass an empty list; ExactOperator should be injected automatically!
    result = run_benchmark(grid, sample, [])

    assert isinstance(result, BenchmarkResult)
    assert "ExactOperator" in result.rmse_wavefield

    # The Exact operator's error against itself should always be exactly 0
    assert result.rmse_wavefield["ExactOperator"] == 0.0


def test_run_benchmark_calculates_rmse():
    """
    This test checks if the RMSE is correctly calculated for an approximate operator.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=4, Nz=2)
    sample = DummySample()

    # This will run the solver and calculate RMSE against the auto-injected ExactOperator
    result = run_benchmark(grid, sample, [DummyApproxOperator])

    assert "DummyApproxOperator" in result.rmse_wavefield

    calculated_error = result.rmse_wavefield["DummyApproxOperator"]

    assert isinstance(calculated_error, float), "Error should be a float number"
    assert calculated_error > 0.0, "The DummyApprox operator should have an error > 0"
