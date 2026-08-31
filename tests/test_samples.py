import pytest
import numpy as np
from unittest.mock import patch

from ptychobench.grid import SimulationGrid
from ptychobench.utils import resolve_backend, to_numpy
from ptychobench.samples import (
    Apoferritin,
    SharpStraightWaveguides,
    StraightBalls,
    StraightWaveguides,
    ZigBalls,
    ZigWaveguides,
)


# --- Adjusted MockGrid to actually trigger boundary conditions ---
class MockGrid:
    def __init__(self):
        self.L = 100.0  # Boundary is at +/- 50.0
        self.N = 128
        self.z_prop = 50.0
        self.Nz = 10
        self.dx = 120.0 / 127
        # Include points beyond +/- 50.0 to trigger the mask
        self.x = np.linspace(-60.0, 60.0, 128)
        # A sample builds in the grid's backend, so the mock has to declare one
        # too. Taken from resolve_backend rather than hard-coded, so the mock
        # cannot drift out of step with what a real grid exposes.
        spec = resolve_backend("numpy")
        self.xp = spec.xp
        self.resolved_device = spec.device
        self.real_dtype = spec.real
        self.complex_dtype = spec.complex


@pytest.fixture
def grid():
    return MockGrid()


# --- Tests ---


def test_waveguide_geometry(grid):
    """Test that StraightWaveguides produces peaks at the correct period."""
    # Set modulus to 0.8 so the peak is clearly > 0.5
    sample = StraightWaveguides(modulus=0.8)
    eps = sample.get_permittivity(grid, z=0.0)

    # Grid center is index 64 (x approx 0)
    # 0.8 * exp(0) = 0.8. 0.8 > 0.5 check passes.
    assert np.abs(eps[64]) > 0.5


def test_boundary_mask(grid):
    """Verify that the boundaries are forced to zero regardless of sample."""
    sample = StraightWaveguides(modulus=1.0)
    eps = sample.get_permittivity(grid, z=0.0)

    # eps[0] corresponds to x=-60 (outside L/2=50), so it must be 0j
    assert eps[0] == 0j
    # eps[-1] corresponds to x=60 (outside L/2=50), so it must be 0j
    assert eps[-1] == 0j


# ---------------------------------------------------------------------------
# The Sample contract, checked for every shipped sample.
#
# These use the real SimulationGrid rather than MockGrid above: the mock sets
# x = linspace(-60, 60) with L = 100, a combination the real grid cannot
# produce, so it exercises a geometry no caller ever sees.
# ---------------------------------------------------------------------------

ALL_SAMPLES = [
    StraightWaveguides,
    SharpStraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
    Apoferritin,
]


@pytest.fixture
def real_grid():
    return SimulationGrid(N=64, Nz=8, L=20.0, z_prop=10.0)


@pytest.mark.parametrize("Cls", ALL_SAMPLES, ids=lambda c: c.__name__)
def test_every_sample_returns_a_complex_slice_of_the_grid_width(Cls, real_grid):
    """The one contract Sample declares, checked for all six implementations."""
    eps = Cls().get_permittivity(real_grid, z=real_grid.z_prop / 3)

    assert eps.shape == (real_grid.N,)
    assert np.iscomplexobj(eps)
    assert np.all(np.isfinite(eps))


@pytest.mark.parametrize("Cls", ALL_SAMPLES, ids=lambda c: c.__name__)
def test_the_domain_edge_is_free_space(Cls, real_grid):
    """Pins _apply_boundary_mask, which every subclass currently calls for
    itself, before that call moves into the base class.

    grid.x is half-open, so x[0] = -L/2 is the only sample the mask reaches.
    """
    eps = Cls().get_permittivity(real_grid, z=0.0)

    assert eps[0] == 0j


@pytest.mark.parametrize("Cls", ALL_SAMPLES, ids=lambda c: c.__name__)
def test_max_gradient_is_finite_and_non_negative(Cls, real_grid):
    """Sample's only non-plotting behaviour of its own; pinned here because it
    is about to be the only thing left on the base class besides the template."""
    grad = Cls().max_gradient(real_grid)

    assert np.isfinite(grad)
    assert grad >= 0.0


@patch("ptychobench.samples.np.load")
def test_apoferritin_loading(mock_load, grid):
    """Test that Apoferritin handles the data array correctly."""
    # Mock a 10x10 array of ones
    mock_data = np.ones((10, 565))
    mock_load.return_value = mock_data

    sample = Apoferritin(modulus=0.1)

    # Ensure it returns an array of the right size
    eps = sample.get_permittivity(grid, z=0.0)
    assert eps.shape == (grid.N,)
    # Verify it is complex
    assert np.iscomplexobj(eps)


# ---------------------------------------------------------------------------
# Backends
#
# A sample reads grid.x and hands its result to an operator, so it has to
# build in whatever backend the grid was made with. Checked at the public
# boundary -- get_permittivity -- rather than per builder.
# ---------------------------------------------------------------------------

pytest.importorskip("torch", reason="the torch backend is an optional extra")


@pytest.mark.parametrize("Cls", ALL_SAMPLES, ids=lambda c: c.__name__)
def test_every_sample_builds_in_the_grids_backend(Cls):
    """The permittivity must come back on the grid's device, whatever the
    sample did internally -- SharpStraightWaveguides goes through SciPy and
    Apoferritin comes off disk, so both are host arrays before the cast."""
    grid = SimulationGrid(N=64, Nz=8, L=20.0, z_prop=10.0, backend="torch")

    eps = Cls().get_permittivity(grid, z=grid.z_prop / 3)

    assert eps.dtype == grid.complex_dtype
    # Compared against grid.x rather than against grid.resolved_device, which
    # is the *name* torch was asked for ("mps") and not the resolved device an
    # array reports ("mps:0"). Same-device-as-the-coordinates is the contract.
    assert eps.device == grid.x.device
    assert eps.shape == (grid.N,)


@pytest.mark.parametrize("Cls", ALL_SAMPLES, ids=lambda c: c.__name__)
def test_every_sample_agrees_across_backends(Cls):
    """Same geometry either way. Compared on the host at single-precision
    tolerance, since the torch grid may resolve to a float32 device."""

    def permittivity(backend: str) -> np.ndarray:
        grid = SimulationGrid(N=64, Nz=8, L=20.0, z_prop=10.0, backend=backend)
        return to_numpy(Cls().get_permittivity(grid, z=grid.z_prop / 3))

    np.testing.assert_allclose(permittivity("torch"), permittivity("numpy"), atol=1e-6)


def test_the_blurred_profile_survives_the_host_round_trip():
    """SharpStraightWaveguides is the one builder that cannot stay on the
    device: scipy.ndimage.gaussian_filter1d is host-only. Pin that the blur is
    still applied, and still lands back in the grid's backend."""
    grid = SimulationGrid(N=64, Nz=8, L=20.0, z_prop=10.0, backend="torch")
    sharp = SharpStraightWaveguides(num_cores=4, core_width=2.0, blur=0.0)
    blurred = SharpStraightWaveguides(num_cores=4, core_width=2.0, blur=3.0)

    edges = np.abs(np.diff(to_numpy(sharp.get_permittivity(grid, 0.0))))
    smoothed = np.abs(np.diff(to_numpy(blurred.get_permittivity(grid, 0.0))))

    assert blurred.get_permittivity(grid, 0.0).dtype == grid.complex_dtype
    assert smoothed.max() < edges.max()


# ---------------------------------------------------------------------------
# Plotting
#
# These three moved out to ptychobench.plotting; Sample keeps one-line
# delegators. One smoke test over the move -- the figures' contents are not a
# contract anyone depends on, but "it draws and returns a figure" is.
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
# Imported explicitly rather than reached as `matplotlib.pyplot`: pyplot is a
# submodule, so that attribute only resolves once something has imported it,
# and until samples.py stopped doing so at module scope, this file was getting
# it for free from the import under test.
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.mark.parametrize(
    "draw",
    [
        lambda s, g: s.plot(g),
        lambda s, g: s.plot_cross_section_modulus(g),
        lambda s, g: s.plot_cross_section_modulus(g, fourier=True),
        lambda s, g: s.plot_cross_section_phase(g),
    ],
)
def test_the_sample_plotters_return_a_figure(draw, real_grid, monkeypatch):
    from matplotlib.figure import Figure

    # The delegators call plt.show(), which Agg warns about since it cannot
    # display anything. Stub it out: what is under test is the returned figure.
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    assert isinstance(draw(StraightWaveguides(), real_grid), Figure)
    plt.close("all")


def test_importing_samples_does_not_pull_in_matplotlib():
    """The three delegators defer `from . import plotting` precisely so that
    someone who only wants a permittivity array does not pay for Matplotlib.
    A module-scope `from matplotlib import pyplot` sat above them for the
    plt.show() calls and silently cost exactly that.

    A subprocess because the assertion is about a *fresh* interpreter: by the
    time this file runs, half the suite has imported pyplot already.
    """
    import subprocess
    import sys

    probe = "import sys, ptychobench.samples; sys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0
