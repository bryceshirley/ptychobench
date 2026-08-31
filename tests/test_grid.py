import dataclasses

import numpy as np
import pytest

from ptychobench.grid import SimulationGrid
from ptychobench.utils import to_numpy


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


def test_the_dft_matrices_are_not_built_until_they_are_asked_for():
    """The O(N^2) matrices are the grid's only expensive objects, and the
    matrix-free paths never touch them. Constructing a grid must stay cheap."""
    grid = SimulationGrid(N=32)

    assert "F" not in vars(grid)
    assert "F_inv" not in vars(grid)

    grid.F

    assert "F" in vars(grid)


def test_the_dft_matrices_are_built_once_per_grid():
    """Laziness must not turn into rebuilding the matrix on every access: the
    dense operator builders read F and F_inv several times each."""
    grid = SimulationGrid(N=32)

    assert grid.F is grid.F
    assert grid.F_inv is grid.F_inv


def test_index_of_x_lands_on_the_nearest_column():
    grid = SimulationGrid(N=8, L=8.0)

    # x = [-4, -3, -2, -1, 0, 1, 2, 3]; half-open, so +4 is not a column.
    assert grid.index_of_x(0.0) == 4
    assert grid.index_of_x(-4.0) == 0
    assert grid.index_of_x(1.4) == 5
    assert grid.index_of_x(1.6) == 6
    # The upper edge is inside the domain but past the last column, so it
    # resolves to that column rather than wrapping around to the first.
    assert grid.index_of_x(4.0) == 7


def test_index_of_x_rejects_a_position_outside_the_domain():
    """Clamping would hand back a plausible edge slice for a position that was
    never simulated."""
    grid = SimulationGrid(N=8, L=8.0)

    with pytest.raises(ValueError, match="outside the domain"):
        grid.index_of_x(4.5)
    with pytest.raises(ValueError, match="outside the domain"):
        grid.index_of_x(-4.5)


@pytest.mark.parametrize("N", [8, 9])
def test_kx_shifted_is_exactly_what_fftshift_produces(N):
    """Both parities: fftshift splits an even axis evenly and an odd one not,
    and kx_shifted derives the ordering with arange instead of calling fftshift
    (fft is an optional Array API extension), so the two have to be checked to
    agree rather than assumed to."""
    grid = SimulationGrid(N=N, L=8.0)

    np.testing.assert_allclose(
        to_numpy(grid.kx_shifted), np.fft.fftshift(to_numpy(grid.kx))
    )


def test_kx_shifted_is_ascending_and_covers_the_band():
    grid = SimulationGrid(N=16, L=8.0)
    kx_shifted = to_numpy(grid.kx_shifted)

    assert kx_shifted.shape == (grid.N,)
    assert np.all(np.diff(kx_shifted) > 0)
    # Nyquist is pi/dx, reached on the negative side for an even N.
    assert kx_shifted[0] == pytest.approx(-np.pi / grid.dx)


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_kx_shifted_is_built_in_the_grids_backend(name):
    pytest.importorskip(name)
    grid = SimulationGrid(N=8, L=8.0, backend=name)

    assert grid.kx_shifted.device == grid.kx.device
    assert grid.kx_shifted.dtype == grid.kx.dtype


def test_angular_spectrum_operator_shape_and_type():
    """Test that the Angular Spectrum operator (L_op) initializes correctly."""
    grid = SimulationGrid(divergence_angle=0.0, N=16)
    L_op = grid.get_angular_spectrum_operator()

    assert L_op.shape == (16, 16)
    assert L_op.dtype == complex


def test_the_kinetic_symbol_reproduces_the_dense_kinetic_operator():
    """The symbol is the Fourier diagonal the dense operator is built from.

    The matrix-free path multiplies by the symbol; the dense path conjugates
    diag(symbol) by the DFT. If these two disagree, every later comparison
    between the paths measures the disagreement rather than the physics.
    """
    grid = SimulationGrid(divergence_angle=0.0, N=32)

    rebuilt = grid.F_inv @ np.diag(grid.kinetic_symbol.astype(complex)) @ grid.F

    np.testing.assert_allclose(rebuilt, grid.get_kinetic_operator(), atol=1e-12)


def test_the_angular_spectrum_symbol_reproduces_the_dense_operator():
    grid = SimulationGrid(divergence_angle=0.0, N=32)

    rebuilt = (
        grid.F_inv @ np.diag(grid.angular_spectrum_symbol.astype(complex)) @ grid.F
    )

    np.testing.assert_allclose(
        rebuilt, grid.get_angular_spectrum_operator(), atol=1e-12
    )


def test_the_symbols_clip_evanescent_modes_to_minus_one():
    """Evanescent modes are removed by setting kz = 0, which sends both
    1 + mu and sqrt(1 + mu) to zero, so both symbols hit -1 there."""
    grid = SimulationGrid(divergence_angle=0.0, N=64, L=10.0, lam=1.0)
    evanescent = ~grid.propagating_mask

    assert evanescent.any(), "the test grid must actually have evanescent modes"
    np.testing.assert_allclose(grid.kinetic_symbol[evanescent], -1.0, atol=1e-14)
    np.testing.assert_allclose(
        grid.angular_spectrum_symbol[evanescent], -1.0, atol=1e-14
    )


def test_the_symbols_are_real_and_even_in_kx():
    """Realness and evenness are exactly the conditions that make the
    Fourier-diagonal operators self-adjoint, which the split-step and Krylov
    paths both lean on."""
    grid = SimulationGrid(divergence_angle=0.0, N=32, L=10.0, lam=1.0)

    for symbol in (grid.kinetic_symbol, grid.angular_spectrum_symbol):
        assert not np.iscomplexobj(symbol)
        # fftfreq puts kx[N - i] = -kx[i], so evenness is a reversal identity.
        np.testing.assert_allclose(symbol[1:], symbol[:0:-1], atol=1e-14)


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


# =========================================================
# Backends
# =========================================================
#: Everything the matrix-free path consumes, which must follow `backend`.
FIELD_ARRAYS = [
    "x",
    "kx",
    "propagating_mask",
    "kinetic_symbol",
    "angular_spectrum_symbol",
]


def test_the_grid_defaults_to_numpy_double_precision():
    """The default is the accuracy reference every other backend is judged
    against, so it must not quietly follow whatever hardware is present."""
    grid = SimulationGrid(N=32)

    assert grid.backend == "numpy"
    assert grid.real_dtype == np.float64
    assert grid.complex_dtype == np.complex128
    assert grid.x.dtype == np.float64
    assert grid.get_initial_field().dtype == np.complex128


@pytest.mark.parametrize("name", FIELD_ARRAYS)
def test_the_field_arrays_are_built_in_the_requested_backend(name):
    torch = pytest.importorskip("torch", reason="the torch backend is optional")
    grid = dataclasses.replace(SimulationGrid(N=32, L=10.0, lam=1.0), backend="torch")

    array = getattr(grid, name)

    assert isinstance(array, torch.Tensor)
    assert str(array.device).startswith(str(grid.resolved_device))


def test_the_initial_field_is_built_in_the_requested_backend():
    """psi_0 is where a run's backend is decided in practice: the operators
    infer their namespace, device and precision from the field handed to them."""
    torch = pytest.importorskip("torch", reason="the torch backend is optional")
    grid = dataclasses.replace(SimulationGrid(N=32), backend="torch")

    psi_0 = grid.get_initial_field()

    assert isinstance(psi_0, torch.Tensor)
    assert psi_0.dtype == grid.complex_dtype


@pytest.mark.parametrize("name", FIELD_ARRAYS)
def test_the_field_arrays_agree_across_backends(name):
    """Same physics, different library. The torch path may be float32, so the
    tolerance is scaled to its dtype rather than to float64."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    numpy_grid = SimulationGrid(N=64, L=10.0, lam=1.0)
    torch_grid = dataclasses.replace(numpy_grid, backend="torch")

    tol = 100 * np.finfo(to_numpy(torch_grid.x).dtype).eps

    np.testing.assert_allclose(
        to_numpy(getattr(torch_grid, name)).astype(float),
        np.asarray(getattr(numpy_grid, name), dtype=float),
        atol=tol,
    )


def test_the_symbols_stay_finite_in_single_precision():
    """1 - kx^2/k0^2 can round below zero in float32 for a mode the band limit
    keeps. sqrt() of that is NaN, and it surfaces much later as an unexplained
    Arnoldi breakdown rather than as an error here."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    grid = dataclasses.replace(SimulationGrid(N=256, L=10.0, lam=1.0), backend="torch")

    assert np.all(np.isfinite(to_numpy(grid.kinetic_symbol)))
    assert np.all(np.isfinite(to_numpy(grid.angular_spectrum_symbol)))


def test_the_host_only_arrays_ignore_the_backend():
    """z_steps drives a Python loop, and the dense matrices are fed to
    scipy.linalg.expm/sqrtm. Neither has anything to gain from a GPU."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    grid = dataclasses.replace(SimulationGrid(N=32, Nz=4), backend="torch")

    assert isinstance(grid.z_steps, np.ndarray)
    assert isinstance(grid.F, np.ndarray)
    assert isinstance(grid.get_kinetic_operator(), np.ndarray)
    assert isinstance(grid.get_angular_spectrum_operator(), np.ndarray)


def test_the_device_can_be_pinned_to_the_host():
    """Comparing torch-on-GPU against NumPy conflates two differences at once.
    Pinning torch to the CPU separates the library from the device -- and it is
    double precision there, because FP64 is what Metal lacks, not what torch
    lacks."""
    torch = pytest.importorskip("torch", reason="the torch backend is optional")
    grid = SimulationGrid(N=32, backend="torch", device="cpu")

    assert grid.resolved_device == "cpu"
    assert grid.real_dtype == torch.float64
    assert str(grid.x.device) == "cpu"
