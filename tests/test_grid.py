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


def test_initial_field_with_non_zero_divergence():
    """Test that a non-zero divergence angle introduces an imaginary phase component."""
    grid = SimulationGrid(divergence_angle=15.0, N=64)
    psi_0 = grid.get_initial_field()

    # If divergence > 0, the field should have a non-trivial imaginary part
    assert not np.allclose(np.imag(psi_0), 0.0)


def test_dft_matrices_are_unitary():
    """
    Test that the pre-calculated DFT matrices F and F_inv are inverses
    of each other. F @ F_inv should equal the Identity matrix.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=32)
    identity = np.eye(grid.N, dtype=complex)

    np.testing.assert_allclose(grid.F @ grid.F_inv, identity, atol=1e-10)
    np.testing.assert_allclose(grid.F_inv @ grid.F, identity, atol=1e-10)


def test_angular_spectrum_operator_shape_and_type():
    """Test that the Angular Spectrum operator (L_op) initializes correctly."""
    grid = SimulationGrid(divergence_angle=0.0, N=16)
    L_op = grid.get_angular_spectrum_operator()

    assert L_op.shape == (16, 16)
    assert L_op.dtype == complex


def test_operator_physics_on_flat_field():
    """
    MATH CHECK: Both the kinetic operator and the angular spectrum operator
    represent spatial derivatives (specifically, relationships to the Laplacian).
    If they act on a perfectly flat field (a constant), the result should be 0,
    just like the derivative of a constant is 0.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=32)
    M = grid.get_kinetic_operator()
    L_op = grid.get_angular_spectrum_operator()

    # Create a flat wave field of all 1s
    flat_field = np.ones(grid.N, dtype=complex)

    # Applying the derivative operators to a flat field should yield arrays of zeros
    np.testing.assert_allclose(M @ flat_field, np.zeros(grid.N), atol=1e-10)
    np.testing.assert_allclose(L_op @ flat_field, np.zeros(grid.N), atol=1e-10)
