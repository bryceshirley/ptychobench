import logging

import numpy as np
import pytest

# Adjust imports based on your actual project structure
from ptychobench.benchmark import _FLUSH_EVERY, run_benchmark
from ptychobench.results import BenchmarkResult
from ptychobench.grid import SimulationGrid
from ptychobench.operators import (
    FeitFleckOperator,
    ForwardOperator,
    LinDudaOperator,
    ParaxialOperator,
    YevickThomsonOperator,
)
from ptychobench.samples import Sample


class DummySample(Sample):
    """A simple sample that does nothing, just to satisfy the Sample class interface."""

    def __init__(self):
        self.modulus = 1.0
        self.dummy_param = "test"

    def _profile(self, grid, z):
        return np.zeros(grid.N, dtype=complex)


class DummyApproxOperator(ForwardOperator):
    """A basic operator that leaves the wave untouched, to test the error calculator."""

    display_name = "DummyApprox (Test)"

    def construct_operator(self, E):
        return np.zeros((self.grid.N, self.grid.N), dtype=complex)

    def step(self, E, psi):
        return psi


# --- TESTS ---


def test_run_benchmark_basic_execution():
    """
    Test that the solver runs without crashing and correctly injects
    and calculates the GroundTruthOperator baseline automatically.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=4, Nz=2)
    sample = DummySample()

    # Pass an empty list; GroundTruthOperator should be injected automatically!
    result = run_benchmark(grid, sample, [])

    assert isinstance(result, BenchmarkResult)
    assert "GroundTruthOperator" in result.rmse_wavefield

    # The Exact operator's error against itself should always be exactly 0
    assert result.rmse_wavefield["GroundTruthOperator"] == 0.0


def test_run_benchmark_calculates_rmse():
    """
    This test checks if the RMSE is correctly calculated for an approximate operator.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=4, Nz=2)
    sample = DummySample()

    # This will run the solver and calculate RMSE against the auto-injected GroundTruthOperator
    result = run_benchmark(grid, sample, [DummyApproxOperator])

    assert "DummyApproxOperator" in result.rmse_wavefield

    calculated_error = result.rmse_wavefield["DummyApproxOperator"]

    assert isinstance(calculated_error, float), "Error should be a float number"
    assert calculated_error > 0.0, "The DummyApprox operator should have an error > 0"


class BumpSample(Sample):
    """A weak refractive-index bump, enough to give the operators something to
    disagree about."""

    def _profile(self, grid, z):
        return 0.05 * np.exp(-(grid.x**2) / 8.0)


def test_the_integrator_choice_reaches_the_operators():
    """The flag has to be threaded through construction, not just accepted."""
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    result = run_benchmark(grid, BumpSample(), [FeitFleckOperator], integrator="krylov")

    assert "FeitFleckOperator" in result.rmse_wavefield


def test_the_krylov_benchmark_agrees_with_the_direct_one():
    """Evaluating the same operators a different way must not move the physics.

    This is the end-to-end version of the per-operator agreement tests: it
    covers the propagation loop, the permittivity plumbing and the metrics, not
    just a single step.
    """
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=4)
    sample = BumpSample()

    direct = run_benchmark(grid, sample, [FeitFleckOperator], integrator="direct")
    krylov = run_benchmark(grid, sample, [FeitFleckOperator], integrator="krylov")

    for name, history in direct.wavefield_history.items():
        np.testing.assert_allclose(krylov.wavefield_history[name], history, atol=1e-9)


class _TwinA(ForwardOperator):
    """Deliberately shares a display label with _TwinB, below."""

    display_name = "Twin"

    def construct_operator(self, E):
        return 0.5 * E


class _TwinB(ForwardOperator):
    """Same label as _TwinA, different generator."""

    display_name = "Twin"

    def construct_operator(self, E):
        return 2.0 * E


def test_operators_sharing_a_display_label_still_get_separate_histories():
    """Two operators must never share a history entry.

    Histories used to be keyed by the display name while the errors were keyed
    by the class name. Two operators with the same label collapsed into one
    entry, and both then had their RMSE computed against whichever wrote last
    -- silently, and with a wrong number for at least one of them.
    """
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    result = run_benchmark(grid, BumpSample(), [_TwinA, _TwinB])

    assert len(result.wavefield_history) == 3
    assert result.display_names["_TwinA"] == result.display_names["_TwinB"] == "Twin"
    assert result.rmse_wavefield["_TwinA"] != result.rmse_wavefield["_TwinB"], (
        "distinct generators must give distinct errors"
    )


def test_an_operator_without_a_display_name_is_rejected_at_definition():
    """The old failure mode was a blank plot legend discovered much later."""
    with pytest.raises(TypeError, match="display_name"):

        class _Nameless(ForwardOperator):
            def construct_operator(self, E):
                return E


def test_the_result_dictionaries_share_one_key_space():
    """Every per-operator dictionary is keyed the same way: by class name.

    All six of them, including the reference's own entry. This used to check
    rmse_wavefield alone, and max_error_detector was in fact missing the
    reference key -- the one dictionary of the three the test did not reach.
    """
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    result = run_benchmark(grid, BumpSample(), [FeitFleckOperator])

    expected = set(result.wavefield_history)
    assert result.REFERENCE in expected
    assert set(result.display_names) == expected
    assert set(result.rmse_wavefield) == expected
    assert set(result.rmse_detector) == expected
    assert set(result.max_error_detector) == expected


def test_the_builtin_operator_display_names_are_unchanged():
    """Plot legends and saved figures depend on these exact strings."""
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    result = run_benchmark(
        grid,
        BumpSample(),
        [
            ParaxialOperator,
            FeitFleckOperator,
            YevickThomsonOperator,
            LinDudaOperator,
        ],
    )

    assert set(result.display_names.values()) == {
        "Ground Truth",
        "Paraxial",
        "Feit/Fleck",
        "Yevick/Thomson",
        "Lin/Duda",
    }


def test_split_step_falls_back_to_krylov_for_the_reference_alone(caplog):
    """GroundTruthOperator is always in the run and has no split-step form.

    Rejecting the run outright would make "split-step" unreachable through
    run_benchmark in every case, so the reference is evaluated with Krylov
    instead. The fallback is a real change of method, so it is logged rather
    than made silently.

    The reference is now the *only* one that falls back. H3 and H4 used to as
    well -- they have no two-factor splitting -- and reach "split-step" through
    a third, series-evaluated factor; that they are absent from the log is what
    says the benchmark noticed.
    """
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    with caplog.at_level(logging.INFO, logger="ptychobench.benchmark"):
        result = run_benchmark(
            grid,
            BumpSample(),
            [FeitFleckOperator, YevickThomsonOperator, LinDudaOperator],
            integrator="split-step",
        )

    for name in ("FeitFleckOperator", "YevickThomsonOperator", "LinDudaOperator"):
        assert name in result.rmse_wavefield
        assert name not in caplog.text
    assert "GroundTruthOperator" in result.rmse_wavefield
    assert "GroundTruthOperator has no split-step form" in caplog.text


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def test_run_benchmark_produces_host_histories_on_any_backend():
    """The histories are the run's record: host NumPy complex128 whatever the
    run executed on, so metrics and plots do not have to care, and a benchmark
    number does not change precision with the hardware it landed on."""
    pytest.importorskip("torch", reason="the torch backend is an optional extra")
    grid = SimulationGrid(N=32, L=10.0, lam=1.0, z_prop=1.0, Nz=4, backend="torch")

    result = run_benchmark(grid, DummySample(), [ParaxialOperator], "krylov")

    assert isinstance(result.sample_history, np.ndarray)
    assert result.sample_history.dtype == np.complex128
    for history in result.wavefield_history.values():
        assert isinstance(history, np.ndarray)
        assert history.dtype == np.complex128


def test_run_benchmark_runs_on_torch_with_the_default_integrator():
    """The default is "direct", and the default is what a first run uses.

    It is host-only -- scipy's expm takes a dense matrix -- but "host-only"
    has to mean "transferred for you", not "raises". Before this was pinned,
    `SimulationGrid(backend="torch")` plus the documented five-step recipe
    crashed on Apple Silicon with a bare torch message about `.cpu()`.
    """
    pytest.importorskip("torch", reason="the torch backend is an optional extra")
    grid = SimulationGrid(N=32, L=10.0, lam=1.0, z_prop=1.0, Nz=4, backend="torch")

    result = run_benchmark(grid, DummySample(), [ParaxialOperator])

    assert np.all(np.isfinite(result.wavefield_history["ParaxialOperator"]))


def test_run_benchmark_agrees_across_backends():
    """One word changes -- backend="torch" -- and the physics does not.

    Pinned on torch:cpu rather than the default device, which on Apple Silicon
    is MPS and therefore float32; that comparison belongs in a tolerance scaled
    to single precision, and is what examples/backend_demo.py reports.
    """
    pytest.importorskip("torch", reason="the torch backend is an optional extra")
    ops = [ParaxialOperator, FeitFleckOperator]

    def histories(backend: str, device=None) -> dict[str, np.ndarray]:
        grid = SimulationGrid(
            N=32, L=10.0, lam=1.0, z_prop=1.0, Nz=4, backend=backend, device=device
        )
        return run_benchmark(grid, DummySample(), ops, "krylov").wavefield_history

    torch_run, numpy_run = histories("torch", "cpu"), histories("numpy")

    assert torch_run.keys() == numpy_run.keys()
    for name in numpy_run:
        np.testing.assert_allclose(torch_run[name], numpy_run[name], atol=1e-12)


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------


def test_the_recorded_trajectory_is_unaffected_by_the_flush_boundary():
    """The history is buffered on the device and copied a block at a time, so a
    run longer than one block has to stitch those blocks back together.

    Nothing in the suite ran past `_FLUSH_EVERY` before this, so the second
    block and the offset it is written at were never exercised at all -- and a
    stitching bug there would misplace whole rows rather than perturb them,
    which every convergence test in the package would sail straight past.

    Pinned against an independently stepped trajectory rather than against the
    other rows of the same history, so it also holds the "record before
    stepping" convention that `plot_evolution_1D` reads.
    """
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=_FLUSH_EVERY + 6)
    sample = BumpSample()

    history = run_benchmark(grid, sample, [FeitFleckOperator]).wavefield_history[
        "FeitFleckOperator"
    ]

    operator = FeitFleckOperator(grid)
    expected = grid.get_initial_field()
    # Every index that could plausibly be misplaced: the first row, both sides
    # of the boundary, and the short final block.
    checkpoints = {0, _FLUSH_EVERY - 1, _FLUSH_EVERY, _FLUSH_EVERY + 1, grid.Nz - 1}
    for i, z in enumerate(grid.z_steps):
        if i in checkpoints:
            np.testing.assert_allclose(history[i], expected, atol=1e-12)
        expected = operator.step(sample.get_permittivity(grid, z), expected)


def test_the_recorded_sample_is_unaffected_by_the_flush_boundary():
    """The permittivity is buffered on the same schedule and stitched by the
    same offset, so it can be misplaced independently of the wavefields."""
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=_FLUSH_EVERY + 6)
    sample = BumpSample()

    recorded = run_benchmark(grid, sample, [FeitFleckOperator]).sample_history

    for i, z in enumerate(grid.z_steps):
        np.testing.assert_allclose(
            recorded[i], sample.get_permittivity(grid, z), atol=1e-12
        )


def test_a_single_operator_class_is_accepted_without_a_sequence():
    """`operators` is a union, and the bare-class arm is what the quick-start
    examples use. Distinguished by what the argument *is*, so a class is never
    mistaken for an iterable of its members."""
    grid = SimulationGrid(N=16, L=10.0, lam=1.0, z_prop=1.0, Nz=2)

    bare = run_benchmark(grid, BumpSample(), FeitFleckOperator)
    wrapped = run_benchmark(grid, BumpSample(), [FeitFleckOperator])

    assert set(bare.wavefield_history) == set(wrapped.wavefield_history)
    np.testing.assert_array_equal(
        bare.wavefield_history["FeitFleckOperator"],
        wrapped.wavefield_history["FeitFleckOperator"],
    )
