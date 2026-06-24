import pytest
import numpy as np
from ptychobench.results import BenchmarkResult

# --- Mock Objects ---


class MockSimulationGrid:
    """A lightweight dummy grid so we don't have to load the real physics parameters."""

    pass


@pytest.fixture
def sample_result():
    """Provides a standard BenchmarkResult object for use in multiple tests."""
    return BenchmarkResult(
        errors={
            "ExactOperator": 0.0,
            "FeitFleckOperator": 4.5e-5,
            "ParaxialOperator": 1.2e-4,
        },
        sample_name="TestSample",
        sample_params={"modulus": 0.8, "period": 10.0},
        grid=MockSimulationGrid(),
        wavefield_history={"ExactOperator": np.zeros((2, 2), dtype=complex)},
        sample_history=np.zeros((2, 2), dtype=complex),
    )


# --- Tests ---


def test_get_errors(sample_result):
    """Tests that the full error dictionary is returned correctly."""
    errors = sample_result.get_errors()
    assert isinstance(errors, dict)
    assert len(errors) == 3
    assert "ParaxialOperator" in errors


def test_get_error_valid_operator(sample_result):
    """Tests retrieving a specific error for an operator that exists."""
    err = sample_result.get_error("FeitFleckOperator")
    assert err == 4.5e-5


def test_get_error_invalid_operator(sample_result):
    """Tests that requesting a missing operator returns infinity instead of crashing."""
    err = sample_result.get_error("NonExistentOperator")
    assert err == float("inf")


def test_print_summary_standard(sample_result, capsys):
    """
    Tests that print_summary outputs the correct parameters and
    sorts the errors from smallest to largest.
    """
    sample_result.print_summary()

    # capsys captures anything printed to the console during the test
    captured = capsys.readouterr()
    output = captured.out

    # Check for metadata
    assert "Benchmark Summary: TestSample" in output
    assert "modulus" in output
    assert "0.8" in output
    assert "period" in output

    # Check that sorting works (Exact should appear before Paraxial)
    idx_exact = output.find("ExactOperator")
    idx_feit = output.find("FeitFleckOperator")
    idx_paraxial = output.find("ParaxialOperator")

    assert idx_exact != -1
    assert idx_paraxial != -1

    # Assert the printed order is Exact -> FeitFleck -> Paraxial
    assert idx_exact < idx_feit < idx_paraxial


def test_print_summary_no_errors(capsys):
    """Tests the fallback print statement when the errors dictionary is empty."""
    empty_result = BenchmarkResult(
        errors={},
        sample_name="EmptySample",
        sample_params={},
        grid=MockSimulationGrid(),
        wavefield_history={},
        sample_history=np.zeros((2, 2), dtype=complex),
    )

    empty_result.print_summary()
    captured = capsys.readouterr()

    assert "No errors calculated (ExactOperator missing from run)." in captured.out
