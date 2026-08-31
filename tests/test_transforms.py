"""Tests for the unitary FFT layer underneath the matrix-free path.

The point of these tests is to pin the matrix-free transform to the dense DFT
matrix the reference path uses. If the two ever disagree on normalisation or
sign convention, every downstream comparison against the dense operator is
measuring the wrong thing.
"""

import numpy as np
import pytest
from array_api_compat import array_namespace

from ptychobench.numerics.apertures import antialias_mask
from ptychobench.grid import SimulationGrid
from ptychobench.numerics.transforms import fft, fourier_multiply, ifft
from ptychobench.utils import resolve_backend, to_numpy


@pytest.fixture
def grid() -> SimulationGrid:
    return SimulationGrid(N=32)


@pytest.fixture
def psi(grid) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(grid.N) + 1j * rng.standard_normal(grid.N)


# --- Agreement with the dense reference ---


def test_fft_matches_the_dense_dft_matrix(grid, psi):
    """`fft` must be exactly `grid.F @ psi`, normalisation and sign included."""
    np.testing.assert_allclose(fft(psi), grid.F @ psi, atol=1e-12)


def test_ifft_matches_the_dense_inverse(grid, psi):
    np.testing.assert_allclose(ifft(psi), grid.F_inv @ psi, atol=1e-12)


def test_roundtrip_is_the_identity(psi):
    np.testing.assert_allclose(ifft(fft(psi)), psi, atol=1e-12)


def test_fft_is_unitary(psi):
    """Parseval. A unitary transform is what makes norm conservation testable."""
    assert float(np.linalg.norm(fft(psi))) == pytest.approx(float(np.linalg.norm(psi)))


# --- Fourier-diagonal operator application ---


def test_fourier_multiply_matches_the_dense_conjugation(grid, psi):
    """`fourier_multiply` is `F_inv @ diag(m) @ F`, which is how grid.py builds
    every Fourier-diagonal operator."""
    multiplier = np.exp(-((grid.kx / grid.k0) ** 2))
    dense = grid.F_inv @ np.diag(multiplier.astype(complex)) @ grid.F

    np.testing.assert_allclose(
        fourier_multiply(psi, multiplier), dense @ psi, atol=1e-12
    )


def test_fourier_multiply_with_a_real_even_multiplier_is_self_adjoint(grid, psi):
    """<u, Av> == <Au, v>, tested rather than assumed.

    The band-limiting mask is the multiplier that matters here: it is real and
    even in kx, which is precisely the condition making the composed operator
    self-adjoint.
    """
    rng = np.random.default_rng(1)
    u = rng.standard_normal(grid.N) + 1j * rng.standard_normal(grid.N)
    mask = antialias_mask(grid.kx)

    lhs = np.vdot(u, fourier_multiply(psi, mask))
    rhs = np.vdot(fourier_multiply(u, mask), psi)

    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_fourier_multiply_by_the_binary_mask_is_idempotent(grid, psi):
    """A projection in Fourier space stays a projection in real space."""
    mask = antialias_mask(grid.kx)
    once = fourier_multiply(psi, mask)
    twice = fourier_multiply(once, mask)
    np.testing.assert_allclose(twice, once, atol=1e-12)


# --- Transverse rank ---
#
# The whole field is transverse: nothing in this package carries a batch axis,
# so a rank-2 array is a 2D transverse field and not a stack of 1D ones. These
# tests pin that reading, because the alternative -- transforming only the last
# axis -- is what the module used to do and is silently wrong for a 2D field
# rather than an error.


def test_fft_transforms_every_axis_of_a_2d_field():
    rng = np.random.default_rng(2)
    psi = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))

    np.testing.assert_allclose(fft(psi), np.fft.fft2(psi, norm="ortho"), atol=1e-12)


def test_the_2d_roundtrip_is_the_identity():
    rng = np.random.default_rng(3)
    psi = rng.standard_normal((8, 16)) + 1j * rng.standard_normal((8, 16))
    np.testing.assert_allclose(ifft(fft(psi)), psi, atol=1e-12)


def test_fourier_multiply_applies_a_2d_symbol():
    """The 2D analogue of the dense-conjugation test.

    A separable symbol is used deliberately: it makes the expected answer
    checkable against two 1D conjugations, so the test does not merely restate
    the implementation with a different spelling of ``fft2``.
    """
    rng = np.random.default_rng(4)
    psi = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    kx = np.fft.fftfreq(8)
    symbol = np.exp(-(kx[:, None] ** 2)) * np.exp(-(kx[None, :] ** 2))

    got = fourier_multiply(psi, symbol)

    # F_inv diag(g) F applied along each axis in turn, which is what a
    # separable symbol means.
    dense = np.fft.ifft(
        np.exp(-(kx**2))[:, None] * np.fft.fft(psi, axis=0, norm="ortho"),
        axis=0,
        norm="ortho",
    )
    dense = np.fft.ifft(
        np.exp(-(kx**2))[None, :] * np.fft.fft(dense, axis=1, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    np.testing.assert_allclose(got, dense, atol=1e-12)


def test_an_explicit_axes_argument_restricts_the_transform(grid, psi):
    """The escape hatch for a caller that does have a batch axis.

    Stacking the same 1D field twice and transforming only the last axis must
    reproduce the 1D answer on each row.
    """
    stacked = np.stack([psi, 2.0 * psi])

    got = fft(stacked, axes=(-1,))

    np.testing.assert_allclose(got[0], grid.F @ psi, atol=1e-12)
    np.testing.assert_allclose(got[1], grid.F @ (2.0 * psi), atol=1e-12)


def test_fft_is_unitary_in_2d():
    """Parseval again, because norm conservation is what the 2D splitting path
    will be tested on and it rests on this."""
    rng = np.random.default_rng(5)
    psi = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    assert float(np.linalg.norm(fft(psi))) == pytest.approx(float(np.linalg.norm(psi)))


# --- Backend agnosticism ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_transforms_match_across_backends(grid, psi, backend_name):
    pytest.importorskip("torch", reason="the torch backend is optional")
    backend = resolve_backend(backend_name)
    psi_backend = backend.xp.asarray(psi, dtype=backend.complex, device=backend.device)

    got = to_numpy(fft(psi_backend))

    # Set the tolerance from the dtype: the MPS path is single precision.
    eps = float(np.finfo(got.dtype).eps)
    np.testing.assert_allclose(got, grid.F @ psi, atol=eps * grid.N * 10)


# --- The optional-extension guard ---


def test_a_namespace_without_fft_raises_a_clear_error(psi, monkeypatch):
    """`fft` is an optional extension in the standard, so its absence is a
    supported state of the world and must not surface as AttributeError."""
    xp = array_namespace(psi)
    monkeypatch.delattr(xp, "fft", raising=True)

    with pytest.raises(NotImplementedError, match="fft"):
        fft(psi)
