"""The matrix-free core on a 2D transverse grid.

Everything the matrix-free path is built from -- the unitary transform, Arnoldi,
and the Lie-Trotter splitting -- is written in terms of elementwise products and
inner products, none of which care about the transverse rank of the field. This module
holds that claim to account on an actual 2D grid rather than leaving it as an
assertion in a docstring.

The dense reference is built by flattening. For a separable unitary DFT,

    fft2(A).flatten() == kron(F, F) @ A.flatten()

in row-major order, so a 2D Fourier-diagonal operator is
``kron(F, F)^H @ diag(symbol.flatten()) @ kron(F, F)`` as an ordinary matrix and
``scipy.linalg.expm`` applies to it exactly as it does in 1D. The grid is kept
at 8x8 so that matrix is 64x64.

Only the tests carrying a dense reference actually discriminate. Against the
previous last-axis-only transform, the five comparisons against ``kron(F, F)``
fail and the other six still pass: norm conservation, self-adjointness and
Krylov-versus-splitting agreement hold for *any* consistent unitary transform,
including the wrong one. They are kept because they pin properties the dense
comparisons do not, but a reader should not mistake them for the load-bearing
ones here.

Scope
-----
:class:`ptychobench.grid.SimulationGrid` and the operator classes on top of it
are 1D-transverse by construction: the grid carries ``x``, ``kx`` and a dense
``N x N`` DFT matrix, and ``GroundTruthOperator`` forms ``sqrt(1 + mu + eps)`` as a
dense matrix on purpose -- that is the reference the whole benchmark is defined
against. So these tests drive the integrators directly with 2D symbols rather
than through an operator class. See docs/technical.md for why the exact
reference cannot follow.
"""

import numpy as np
import pytest
import scipy.linalg

from ptychobench.numerics.integrators import krylov_funm
from ptychobench.numerics.splitting import lie_trotter
from ptychobench.numerics.transforms import fft, fourier_multiply
from ptychobench.utils import resolve_backend, to_numpy

#: Small enough that kron(F, F) is a 64x64 dense matrix.
N = 8

#: An imaginary step, so every factor below has unit modulus and the composed
#: propagator is unitary.
STEP = 0.35j


@pytest.fixture
def unitary_dft_2d():
    """kron(F, F) for the unitary 1D DFT F, the flattened form of fft2."""
    F = scipy.linalg.dft(N) / np.sqrt(N)
    return np.kron(F, F)


@pytest.fixture
def kinetic_symbol():
    """A real, even 2D symbol standing in for -kx^2 - ky^2 over k0^2."""
    k = 2.0 * np.pi * np.fft.fftfreq(N)
    return -(k[:, None] ** 2 + k[None, :] ** 2)


@pytest.fixture
def potential():
    """A real 2D refractive potential, so the generator stays self-adjoint."""
    coord = np.linspace(-3.0, 3.0, N)
    r2 = coord[:, None] ** 2 + coord[None, :] ** 2
    return 0.08 * np.exp(-r2 / 4.0)


@pytest.fixture
def psi():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))


def _dense_generator(unitary_dft_2d, kinetic_symbol, potential):
    """L + V as a 64x64 matrix acting on the flattened field."""
    kinetic = (
        unitary_dft_2d.conj().T
        @ np.diag(kinetic_symbol.flatten().astype(complex))
        @ unitary_dft_2d
    )
    return kinetic + np.diag(potential.flatten().astype(complex))


# --- The flattening identity the rest of the module rests on ---


def test_the_2d_transform_is_the_kronecker_product(unitary_dft_2d, psi):
    """If this fails, every dense reference below is comparing against the wrong
    matrix and the other failures would be misleading."""
    np.testing.assert_allclose(
        fft(psi).flatten(), unitary_dft_2d @ psi.flatten(), atol=1e-12
    )


def test_a_2d_fourier_multiply_is_the_dense_conjugation(
    unitary_dft_2d, kinetic_symbol, psi
):
    dense = (
        unitary_dft_2d.conj().T
        @ np.diag(kinetic_symbol.flatten().astype(complex))
        @ unitary_dft_2d
    )
    np.testing.assert_allclose(
        fourier_multiply(psi, kinetic_symbol).flatten(),
        dense @ psi.flatten(),
        atol=1e-12,
    )


# --- Splitting ---


def test_a_2d_split_step_preserves_the_field_shape(kinetic_symbol, potential, psi):
    """The rank must survive the step. A splitting that collapsed the field to
    1D would still produce finite numbers, so shape is worth asserting."""
    assert lie_trotter(psi, kinetic_symbol, potential, STEP).shape == (N, N)


def test_a_2d_free_space_step_is_exact(unitary_dft_2d, kinetic_symbol, psi):
    """With no potential the product collapses to the single kinetic factor,
    which is the analytic angular-spectrum solution in 2D as much as in 1D."""
    got = lie_trotter(psi, kinetic_symbol, np.zeros((N, N)), STEP)

    kinetic = (
        unitary_dft_2d.conj().T
        @ np.diag(kinetic_symbol.flatten().astype(complex))
        @ unitary_dft_2d
    )
    reference = scipy.linalg.expm(STEP * kinetic) @ psi.flatten()
    np.testing.assert_allclose(got.flatten(), reference, atol=1e-11)


def test_2d_lie_trotter_is_first_order_in_the_step(
    unitary_dft_2d, kinetic_symbol, potential, psi
):
    """The order is a property of the product structure, not of the rank, so it
    must come out at 1 here exactly as it does in test_splitting.py."""
    total = 1.0j
    reference = (
        scipy.linalg.expm(
            total * _dense_generator(unitary_dft_2d, kinetic_symbol, potential)
        )
        @ psi.flatten()
    )

    counts = np.array([4, 8, 16, 32])
    errors = []
    for n_steps in counts:
        field = psi
        for _ in range(int(n_steps)):
            field = lie_trotter(field, kinetic_symbol, potential, total / n_steps)
        errors.append(float(np.linalg.norm(field.flatten() - reference)))

    slope, _ = np.polyfit(np.log(1.0 / counts), np.log(np.array(errors)), 1)
    assert 0.9 < slope < 1.2, f"expected global order 1, measured {slope:.3f}"


def test_the_2d_norm_is_conserved_for_a_real_potential(kinetic_symbol, potential, psi):
    field = psi
    for _ in range(40):
        field = lie_trotter(field, kinetic_symbol, potential, STEP)

    assert float(np.linalg.norm(field)) == pytest.approx(
        float(np.linalg.norm(psi)), rel=1e-12
    )


# --- Krylov ---


def test_krylov_matches_the_dense_propagator_in_2d(
    unitary_dft_2d, kinetic_symbol, potential, psi
):
    """Arnoldi only ever needs a matvec and an inner product, so a 2D field is
    just a vector it never flattens. This is the test that says so."""

    def generator(field):
        return STEP * (fourier_multiply(field, kinetic_symbol) + potential * field)

    got = krylov_funm(generator, psi, scipy.linalg.expm, max_iter=60, tol=1e-12)

    reference = (
        scipy.linalg.expm(
            STEP * _dense_generator(unitary_dft_2d, kinetic_symbol, potential)
        )
        @ psi.flatten()
    )
    assert got.shape == (N, N)
    np.testing.assert_allclose(got.flatten(), reference, atol=1e-10)


def test_the_2d_generator_is_self_adjoint(kinetic_symbol, potential, psi):
    """<u, Av> == <Au, v> for a real symbol and a real potential, which is what
    makes the propagator unitary and the norm test above meaningful."""
    rng = np.random.default_rng(1)
    other = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

    def generator(field):
        return fourier_multiply(field, kinetic_symbol) + potential * field

    left = complex(np.vdot(other, generator(psi)))
    right = complex(np.vdot(generator(other), psi))
    assert left == pytest.approx(right, abs=1e-12)


def test_krylov_and_split_step_converge_to_each_other_in_2d(
    kinetic_symbol, potential, psi
):
    """Two independent evaluations of the same 2D generator. Neither is the
    dense reference, so agreement is evidence that both are right rather than
    that both were fitted to the same number."""

    def generator(field):
        return STEP * (fourier_multiply(field, kinetic_symbol) + potential * field)

    krylov = krylov_funm(generator, psi, scipy.linalg.expm, max_iter=60, tol=1e-13)

    split = psi
    # 512 was enough while the splitting was second order; it leaves a 7.2e-06
    # residual at first order. The step count buys the agreement, so it scales
    # with the order rather than the tolerance being relaxed to meet it -- a
    # loosened tolerance here would stop discriminating against a genuinely
    # divergent path.
    n_steps = 4096
    for _ in range(n_steps):
        split = lie_trotter(split, kinetic_symbol, potential, STEP / n_steps)

    relative = float(np.linalg.norm(krylov - split)) / float(np.linalg.norm(krylov))
    assert relative < 1e-6, f"the two paths differ by {relative:.2e}"


# --- Backend agnosticism ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_the_2d_split_step_runs_on_every_backend(
    backend_name, kinetic_symbol, potential, psi
):
    """The MPS path is single precision, so the tolerance comes from the dtype
    rather than from a constant."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    backend = resolve_backend(backend_name)

    def as_backend(array, dtype):
        return backend.xp.asarray(array, dtype=dtype, device=backend.device)

    got = to_numpy(
        lie_trotter(
            as_backend(psi, backend.complex),
            as_backend(kinetic_symbol, backend.real),
            as_backend(potential, backend.real),
            STEP,
        )
    )

    reference = to_numpy(lie_trotter(psi, kinetic_symbol, potential, STEP))
    eps = float(np.finfo(got.dtype).eps)
    assert got.shape == (N, N)
    assert float(np.linalg.norm(got - reference)) < eps**0.5 * float(
        np.linalg.norm(reference)
    )
