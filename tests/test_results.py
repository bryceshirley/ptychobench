import pytest
import numpy as np
from ptychobench.grid import SimulationGrid
from ptychobench.results import BenchmarkResult
from ptychobench.utils import to_numpy

# --- Mock Objects ---


def _small_grid() -> SimulationGrid:
    """A minimal real grid. None of these tests read from it, but BenchmarkResult
    requires one, and a tiny real grid costs less than keeping a mock honest."""
    return SimulationGrid(N=4, Nz=2)


@pytest.fixture
def sample_result():
    """Provides a standard BenchmarkResult object for use in multiple tests."""
    return BenchmarkResult(
        rmse_wavefield={
            "GroundTruthOperator": 0.0,
            "FeitFleckOperator": 4.5e-5,
            "ParaxialOperator": 1.2e-4,
        },
        max_error_detector={
            "GroundTruthOperator": 0.0,
            "FeitFleckOperator": 3.2e-5,
            "ParaxialOperator": 1.0e-4,
        },
        rmse_detector={
            "GroundTruthOperator": 0.0,
            "FeitFleckOperator": 3.2e-5,
            "ParaxialOperator": 1.0e-4,
        },
        sample_name="TestSample",
        sample_params={"modulus": 0.8, "period": 10.0},
        grid=_small_grid(),
        wavefield_history={"GroundTruthOperator": np.zeros((2, 2), dtype=complex)},
        sample_history=np.zeros((2, 2), dtype=complex),
    )


# --- Tests ---


def test_get_errors(sample_result):
    """Tests that the full error dictionary is returned correctly."""
    errors = sample_result.get_rmse_wavefield()
    assert isinstance(errors, dict)
    assert len(errors) == 3
    assert "ParaxialOperator" in errors


def test_get_error_valid_operator(sample_result):
    """Tests retrieving a specific error for an operator that exists."""
    err = sample_result.get_rmse_wavefield("FeitFleckOperator")
    assert err == 4.5e-5


@pytest.mark.parametrize("accessor", ["get_rmse_wavefield", "get_rmse_detector"])
def test_asking_for_an_operator_that_was_not_in_the_run_raises(sample_result, accessor):
    """This used to return float("inf"), which is a wrong number dressed as a
    right one: it formats, sorts and plots like a real error, so a typo in an
    operator name read as "this approximation was infinitely bad". The message
    lists what was actually available, which is the useful half."""
    with pytest.raises(KeyError, match="Paraxailoperator"):
        getattr(sample_result, accessor)("Paraxailoperator")

    with pytest.raises(KeyError, match="ParaxialOperator"):
        getattr(sample_result, accessor)("Paraxailoperator")


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

    # Check that sorting works (Ground Truth should appear before Paraxial)
    idx_exact = output.find("GroundTruthOperator")
    idx_feit = output.find("FeitFleckOperator")
    idx_paraxial = output.find("ParaxialOperator")

    assert idx_exact != -1
    assert idx_paraxial != -1

    # Assert the printed order is Ground Truth -> FeitFleck -> Paraxial
    assert idx_exact < idx_feit < idx_paraxial


def test_print_summary_no_errors(capsys):
    """Tests the fallback print statement when the errors dictionary is empty."""
    empty_result = BenchmarkResult(
        rmse_wavefield={},
        max_error_detector={},
        rmse_detector={},
        sample_name="EmptySample",
        sample_params={},
        grid=_small_grid(),
        wavefield_history={},
        sample_history=np.zeros((2, 2), dtype=complex),
    )

    empty_result.print_summary()
    captured = capsys.readouterr()

    assert (
        "No errors calculated (GroundTruthOperator missing from run)." in captured.out
    )


# ---------------------------------------------------------------------------
# Characterization tests for the plotting and reporting split.
#
# Deliberately few, and all at the public boundary. Asserting on axis labels,
# colours or line counts would triple the count, pin nothing a user depends on,
# and have to be rewritten by the very refactor they exist to protect.
#
# `plot_evolution_slider`'s `_update` callback stays uncovered: reaching it
# needs a synthetic Matplotlib event, which is more harness than one
# interactive helper is worth.
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture
def real_result():
    """One real, small run. The plotters read wavefield_history and the grid
    together, so a hand-built result with mismatched shapes would not exercise
    them."""
    from ptychobench.benchmark import run_benchmark
    from ptychobench.operators import FeitFleckOperator, ParaxialOperator
    from ptychobench.samples import StraightWaveguides

    grid = SimulationGrid(N=32, Nz=4, L=10.0, lam=1.0, z_prop=1.0)
    return run_benchmark(
        grid,
        StraightWaveguides(num_cores=2, core_width=1.0),
        [ParaxialOperator, FeitFleckOperator],
        integrator="krylov",
    )


#: The output contract of ``generate_report``: these filenames are what the
#: Results Guide tells a user to look for.
REPORT_FILES = [
    "simulation_summary.txt",
    "errors.md",
    "propagated_fields_2D_amp.png",
    "propagated_fields_2D_phase.png",
    "phase_evolution_1D.png",
    "amplitude_evolution_1D.png",
    "farfield_error_intensity.png",
    "farfield_error_magnitude.png",
    "farfield_error_phase.png",
]


def test_generate_report_writes_the_documented_file_set(
    real_result, tmp_path, monkeypatch
):
    """The strongest single characterization test available: one call
    exercising most of the module. Pins the filenames, which are the actual
    public contract, and the timestamped-subdirectory layout."""
    monkeypatch.chdir(tmp_path)

    real_result.generate_report(save_dir="results")

    runs = list((tmp_path / "results").iterdir())
    assert len(runs) == 1
    assert runs[0].name.startswith(f"run_{real_result.sample_name}_")

    written = {p.name for p in runs[0].iterdir()}
    assert written == set(REPORT_FILES) | {f"{real_result.sample_name}.png"}


def test_the_report_records_the_grid_and_the_sample(real_result, tmp_path, monkeypatch):
    """The summary and the error table are the two hand-rolled serialisers, and
    the only part of the report a human reads rather than looks at."""
    monkeypatch.chdir(tmp_path)

    real_result.generate_report(save_dir="results")
    run_dir = next((tmp_path / "results").iterdir())

    summary = (run_dir / "simulation_summary.txt").read_text()
    assert f"Sample:    {real_result.sample_name}" in summary
    assert f"N (pixels):  {real_result.grid.N}" in summary

    errors = (run_dir / "errors.md").read_text()
    # The reference is the thing errors are measured against, so it is not a row.
    assert "ParaxialOperator" in errors
    assert "| GroundTruthOperator |" not in errors
    # All three metrics, not the two it used to carry. print_summary and the
    # Results Guide both cover three, so a reader comparing the saved table
    # against either found one silently absent.
    assert "Wavefield RMSE" in errors
    assert "Detector Intensity RMSE" in errors
    assert "Detector Max Abs Error" in errors
    # One value per metric per non-reference operator, i.e. no "—" placeholders.
    assert "—" not in errors


@pytest.mark.parametrize(
    "call, shape",
    [
        (lambda r: r.plot_sample(), "figure"),
        (lambda r: r.plot_wavefields(), "pair"),
        (lambda r: r.plot_evolution_1D(), "pair"),
        (lambda r: r.plot_farfield_error(), "figure"),
        (lambda r: r.plot_evolution_slider(), "fig_and_dict"),
    ],
)
def test_each_plot_method_returns_what_callers_destructure(real_result, call, shape):
    """Pins the return protocol -- fig, (fig, fig), (fig, widgets) -- which is
    what the notebook and the guides unpack. Nothing about the contents."""
    from matplotlib.figure import Figure

    returned = call(real_result)

    if shape == "figure":
        assert isinstance(returned, Figure)
    elif shape == "pair":
        first, second = returned
        assert isinstance(first, Figure) and isinstance(second, Figure)
    else:
        fig, widgets = returned
        assert isinstance(fig, Figure)
        assert "x_slider" in widgets

    matplotlib.pyplot.close("all")


def test_the_farfield_plot_draws_against_the_shifted_frequency_axis(real_result):
    """The far field is indexed by transverse frequency, not by x.

    An exception to the "no asserting on plot contents" note above, because
    this is not cosmetic: the abscissa used to be linspace(-L/2, L/2, N), which
    put a real-space axis under a Fourier-space curve and was off by a factor
    of L*dx/(2*pi) -- eight, at the package defaults. The curve shape was right
    all along, so nothing but reading the numbers off the axis catches it."""
    fig = real_result.plot_farfield_error()
    ax = fig.axes[0]

    expected = np.fft.fftshift(to_numpy(real_result.grid.kx))
    for line in ax.lines:
        np.testing.assert_allclose(line.get_xdata(), expected)

    matplotlib.pyplot.close("all")


#: The three plotters that draw an (Nz, N) history as an image. Each used to
#: write its own extent, and two of the three wrote a wrong one.
MAP_PLOTTERS = {
    "plot_sample": lambda r, s: [r.plot_sample()],
    "plot_wavefields": lambda r, s: list(r.plot_wavefields()),
    "plot_sample_profile": lambda r, s: [s.plot(r.grid)],
}


@pytest.mark.parametrize("draw", MAP_PLOTTERS.values(), ids=MAP_PLOTTERS)
def test_the_map_plots_put_the_first_history_row_at_the_first_z_step(real_result, draw):
    """Row i of a history is the field at z_steps[i]; the image must say so.

    The same exception as the far-field test above -- geometry, not cosmetics.
    `plot_sample_profile` paired an ascending extent with origin="upper", which
    draws the map upside down in z: row 0 sat at the axis position labelled
    z_prop - dz. Nothing but reading the extent catches it, because a flipped
    image is still a plausible-looking image.
    """
    from ptychobench.samples import StraightWaveguides
    from ptychobench.utils import to_numpy

    grid = real_result.grid
    x = to_numpy(grid.x)
    expected = (
        float(x[0]),
        float(x[-1]),
        float(grid.z_steps[-1]),
        float(grid.z_steps[0]),
    )

    for fig in draw(real_result, StraightWaveguides(num_cores=2, core_width=1.0)):
        for image in [im for ax in fig.axes for im in ax.images]:
            assert image.origin == "upper"
            np.testing.assert_allclose(image.get_extent(), expected)

    matplotlib.pyplot.close("all")


@pytest.mark.parametrize("plotter", ["plot_evolution_1D", "plot_evolution_slider"])
def test_the_evolution_plots_reject_an_out_of_bounds_x(real_result, plotter):
    """Documented behaviour, and the only validation either plotter does."""
    with pytest.raises(ValueError, match="out of bounds|outside the domain"):
        getattr(real_result, plotter)(x=real_result.grid.L)


def test_both_evolution_plots_pick_the_same_column(real_result):
    """The static and interactive plots of 'the slice at x' must be the same
    slice. They used to disagree: plot_evolution_1D mapped x through
    int((x + L/2) / L * (N - 1)), which is the formula for a closed linspace,
    while grid.x is half-open -- so the two landed on different columns for
    every x but the left edge."""
    grid = real_result.grid
    x = grid.L / 4

    fig_1d, _ = real_result.plot_evolution_1D(x=x)
    fig_slider, _ = real_result.plot_evolution_slider(x=x)

    def plotted(fig):
        return [line.get_ydata() for ax in fig.axes for line in ax.get_lines()]

    # The slider figure's phase axis holds the same curves the 1D phase figure
    # does, if and only if both resolved x to the same index.
    np.testing.assert_allclose(plotted(fig_1d)[0], plotted(fig_slider)[0])

    matplotlib.pyplot.close("all")


# ---------------------------------------------------------------------------
# Backends
#
# run_benchmark hands back host histories, so most of a result is already
# NumPy. What is not is `result.grid`: its x, kx and field arrays stay on the
# device, and every plotter reads them. Matplotlib cannot take a device tensor,
# so this is the one place a backend can leak into a figure -- worth exercising
# the whole reporting surface once rather than each conversion in isolation.
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_result():
    """The same small run as `real_result`, on a torch grid."""
    pytest.importorskip("torch", reason="the torch backend is an optional extra")
    from ptychobench.benchmark import run_benchmark
    from ptychobench.operators import FeitFleckOperator, ParaxialOperator
    from ptychobench.samples import StraightWaveguides

    grid = SimulationGrid(N=32, Nz=4, L=10.0, lam=1.0, z_prop=1.0, backend="torch")
    return run_benchmark(
        grid,
        StraightWaveguides(num_cores=2, core_width=1.0),
        [ParaxialOperator, FeitFleckOperator],
        integrator="krylov",
    )


def test_a_torch_backed_run_reports_and_plots(torch_result, tmp_path, monkeypatch):
    """One call over the whole reporting surface. `generate_report` draws every
    figure the package can draw, so if any plotter passed a device tensor to
    Matplotlib this is where it would surface."""
    monkeypatch.chdir(tmp_path)

    torch_result.generate_report(save_dir="results")

    run_dir = next((tmp_path / "results").iterdir())
    written = {p.name for p in run_dir.iterdir()}
    assert written == set(REPORT_FILES) | {f"{torch_result.sample_name}.png"}


def test_the_interactive_plotters_work_on_a_torch_grid(torch_result):
    """The two evolution plots are the ones that index the grid rather than the
    history -- `index_of_x` divides by `grid.dx` and the slider steps by it."""
    from matplotlib.figure import Figure

    fig, widgets = torch_result.plot_evolution_slider(x=torch_result.grid.L / 4)

    assert isinstance(fig, Figure)
    assert "x_slider" in widgets
    matplotlib.pyplot.close("all")
