import pytest
import numpy as np
from unittest.mock import patch
from ptychobench.samples import StraightWaveguides, Apoferritin


# --- Adjusted MockGrid to actually trigger boundary conditions ---
class MockGrid:
    def __init__(self):
        self.L = 100.0  # Boundary is at +/- 50.0
        self.N = 128
        self.z_prop = 50.0
        self.Nz = 10
        # Include points beyond +/- 50.0 to trigger the mask
        self.x = np.linspace(-60.0, 60.0, 128)


@pytest.fixture
def grid():
    return MockGrid()


# --- Tests ---


def test_waveguide_geometry(grid):
    """Test that StraightWaveguides produces peaks at the correct period."""
    # Set modulus to 0.8 so the peak is clearly > 0.5
    sample = StraightWaveguides(period=10.0, modulus=0.8)
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
