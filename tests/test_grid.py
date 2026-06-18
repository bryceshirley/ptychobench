import numpy as np
from ptychobench.grid import SimulationGrid


def test_grid_initialization_and_shapes():
    """Test that the grid initializes derived properties and shapes correctly."""
    grid = SimulationGrid(divergence_angle=10.0, N=64, Nz=20, L=10.0)

    # 1. Check array shapes
    assert grid.x.shape == (64,)
    assert grid.kx.shape == (64,)
    assert grid.z_steps.shape == (20,)

    # 2. Check derived properties
    assert grid.dx == 10.0 / 64
    assert np.isclose(grid.k0, 2 * np.pi / grid.lam)

    # 3. Check matrix generation shapes
    M = grid.get_kinetic_operator()
    assert M.shape == (64, 64)
    assert M.dtype == complex


def test_initial_field_divergence():
    """Test that divergence angle = 0 results in a flat phase profile."""
    grid_flat = SimulationGrid(divergence_angle=0.0, N=64)
    psi_0 = grid_flat.get_initial_field()

    # If divergence is 0, the phase should be 0, meaning the array is purely real
    # (ignoring standard floating point errors)
    assert np.allclose(np.imag(psi_0), 0.0)
