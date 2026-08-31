"""Tests for the matrix-free evaluation of f(A)v by Krylov projection.

The dense path is the accuracy reference throughout: every accuracy claim here
is checked against `scipy.linalg.expm` / `sqrtm` applied to the full matrix,
not against a previously recorded number.
"""

import warnings

import numpy as np
import pytest
import scipy.linalg

from ptychobench.numerics.integrators import (
    arnoldi,
    krylov_funm,
    warn_if_near_branch_cut,
)
from ptychobench.utils import resolve_backend, to_numpy


def _matvec(A):
    """Wrap a dense matrix as the matrix-free callable the integrators take."""

    def multiply(x):
        return A @ x

    return multiply


@pytest.fixture
def hermitian():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((24, 24)) + 1j * rng.standard_normal((24, 24))
    return M + M.conj().T


@pytest.fixture
def non_normal():
    """Upper triangular with a spread spectrum: A A^H != A^H A by a long way.

    Non-normal operators are the case the first-order stopping estimate is
    weakest on, which is exactly why one is here.
    """
    rng = np.random.default_rng(1)
    M = np.triu(rng.standard_normal((24, 24)) + 1j * rng.standard_normal((24, 24)))
    np.fill_diagonal(M, np.linspace(0.5, 4.0, 24))
    return M


@pytest.fixture
def v():
    rng = np.random.default_rng(2)
    return rng.standard_normal(24) + 1j * rng.standard_normal(24)


# --- Agreement with the dense reference ---


def test_krylov_expm_matches_the_dense_reference_on_a_hermitian_operator(hermitian, v):
    got = krylov_funm(_matvec(hermitian), v, scipy.linalg.expm, max_iter=24, tol=1e-12)
    np.testing.assert_allclose(got, scipy.linalg.expm(hermitian) @ v, atol=1e-9)


def test_krylov_expm_matches_the_dense_reference_on_a_non_normal_operator(
    non_normal, v
):
    got = krylov_funm(_matvec(non_normal), v, scipy.linalg.expm, max_iter=24, tol=1e-12)
    np.testing.assert_allclose(got, scipy.linalg.expm(non_normal) @ v, atol=1e-9)


def test_krylov_sqrtm_matches_the_dense_reference(hermitian, v):
    """A positive definite operator, so the principal square root is unambiguous."""
    spd = hermitian @ hermitian.conj().T + 4 * np.eye(24)
    got = krylov_funm(_matvec(spd), v, scipy.linalg.sqrtm, max_iter=24, tol=1e-12)
    np.testing.assert_allclose(got, scipy.linalg.sqrtm(spd) @ v, atol=1e-8)


def test_a_full_dimension_subspace_is_exact(hermitian, v):
    """With m = N the projection is onto the whole space, so only round-off
    separates the Krylov answer from the dense one."""
    got = krylov_funm(_matvec(hermitian), v, scipy.linalg.expm, max_iter=24, tol=0.0)
    np.testing.assert_allclose(got, scipy.linalg.expm(hermitian) @ v, atol=1e-10)


def test_a_larger_subspace_is_never_worse(hermitian, v):
    reference = scipy.linalg.expm(hermitian) @ v
    errors = [
        float(
            np.linalg.norm(
                krylov_funm(
                    _matvec(hermitian), v, scipy.linalg.expm, max_iter=m, tol=0.0
                )
                - reference
            )
        )
        for m in (4, 8, 16)
    ]
    assert errors[0] > errors[1] > errors[2]


# --- The Arnoldi recurrence itself ---


def test_the_arnoldi_relation_holds(hermitian, v):
    """A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T, the identity everything rests on."""
    result = arnoldi(_matvec(hermitian), v, max_iter=10)
    m = result.size
    V = result.basis[:m, :]
    H = result.hessenberg[:m, :m]

    lhs = hermitian @ V.T
    rhs = V.T @ H
    rhs[:, -1] += result.hessenberg[m, m - 1] * result.basis[m, :]

    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_the_basis_is_orthonormal_to_machine_precision(non_normal, v):
    """This is the test that distinguishes CGS2 from a single Gram-Schmidt pass.

    One pass loses orthogonality at around sqrt(eps); applying classical
    Gram-Schmidt twice recovers it to eps. Anything looser than the bound below
    means the second pass is missing.
    """
    result = arnoldi(_matvec(non_normal), v, max_iter=20)
    V = result.basis[: result.size, :]
    gram = V.conj() @ V.T

    departure = float(np.max(np.abs(gram - np.eye(result.size))))
    assert departure < 200 * np.finfo(np.complex128).eps


def test_the_first_basis_vector_is_v_normalised(hermitian, v):
    result = arnoldi(_matvec(hermitian), v, max_iter=5)
    assert result.beta == pytest.approx(np.linalg.norm(v))
    np.testing.assert_allclose(result.basis[0, :], v / np.linalg.norm(v), atol=1e-12)


def test_the_hessenberg_matrix_is_upper_hessenberg(hermitian, v):
    result = arnoldi(_matvec(hermitian), v, max_iter=8)
    H = result.hessenberg[: result.size, : result.size]
    np.testing.assert_allclose(np.tril(H, -2), 0.0, atol=1e-14)


# --- Happy breakdown: an invariant subspace, not a failure ---


def test_happy_breakdown_stops_early_and_is_exact():
    """When the subspace becomes A-invariant, f(A)v lies inside it exactly.

    A is block diagonal and v lives entirely in the first 3x3 block, so the
    Krylov space cannot grow past dimension 3 -- and the answer there is not an
    approximation, it is the answer.
    """
    rng = np.random.default_rng(3)
    small = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    other = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    A = scipy.linalg.block_diag(small, other)

    v = np.zeros(8, dtype=complex)
    v[:3] = [1.0, 2.0, -1.0]

    result = arnoldi(_matvec(A), v, max_iter=8)
    assert result.breakdown
    assert result.size == 3

    got = krylov_funm(_matvec(A), v, scipy.linalg.expm, max_iter=8, tol=0.0)
    np.testing.assert_allclose(got, scipy.linalg.expm(A) @ v, atol=1e-12)


def test_breakdown_does_not_divide_by_the_residual_norm():
    """A nilpotent operator drives the residual norm to exactly zero.

    An unguarded normalisation would produce nan/inf here rather than the
    correct finite answer.
    """
    A = np.diag(np.ones(4), k=1)[:5, :5].astype(complex)  # shifts, A^5 == 0
    v = np.zeros(5, dtype=complex)
    v[0] = 1.0

    got = krylov_funm(_matvec(A), v, scipy.linalg.expm, max_iter=10, tol=0.0)

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, scipy.linalg.expm(A) @ v, atol=1e-12)


def test_a_zero_vector_returns_zero_without_dividing_by_its_norm(hermitian):
    zero = np.zeros(24, dtype=complex)
    got = krylov_funm(_matvec(hermitian), zero, scipy.linalg.expm, max_iter=5)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, 0.0, atol=0.0)


# --- The stopping criterion ---


def test_tolerance_is_relative_to_the_norm_of_v(hermitian, v):
    """`tol` is documented as relative to ||v||, so scaling v must not change
    how many iterations are spent."""
    calls_small, calls_large = [], []

    def counting(A, log):
        def apply(x):
            log.append(1)
            return A @ x

        return apply

    krylov_funm(counting(hermitian, calls_small), v, scipy.linalg.expm, tol=1e-8)
    krylov_funm(counting(hermitian, calls_large), 1e6 * v, scipy.linalg.expm, tol=1e-8)

    assert len(calls_small) == len(calls_large)


def test_a_loose_tolerance_stops_sooner_than_a_tight_one(hermitian, v):
    loose = arnoldi(
        _matvec(hermitian),
        v,
        max_iter=24,
        stop=_expm_stop(1e-2),
    )
    tight = arnoldi(
        _matvec(hermitian),
        v,
        max_iter=24,
        stop=_expm_stop(1e-12),
    )
    assert loose.size < tight.size


def _expm_stop(tol):
    def stop(H_active, h_next, step):
        fH = scipy.linalg.expm(H_active)
        return h_next * abs(fH[step - 1, 0]) <= tol

    return stop


def test_an_exception_from_f_is_not_swallowed(hermitian, v):
    """A failure inside f(H) must surface. Catching it would silently disable
    the early exit and quietly burn every remaining iteration."""

    class Boom(Exception):
        pass

    def exploding(_H):
        raise Boom("f(H) failed")

    with pytest.raises(Boom):
        krylov_funm(_matvec(hermitian), v, exploding, max_iter=6)


def test_a_nonfinite_convergence_estimate_is_reported(hermitian, v):
    """A nan estimate compares False against any tolerance, so a silent
    `return False` would look exactly like "not converged yet" forever."""

    def nan_valued(H):
        return np.full_like(H, np.nan)

    with pytest.raises(FloatingPointError, match="estimate"):
        krylov_funm(_matvec(hermitian), v, nan_valued, max_iter=6)


# --- Branch cuts ---


def test_a_negative_real_eigenvalue_triggers_a_branch_cut_warning():
    """sqrt has its branch cut on the negative real axis. Clipping evanescent
    modes drives eigenvalues onto it, where the principal root flips sign for
    an arbitrarily small change in the imaginary part."""
    M = np.diag([1.0 + 0j, -2.0 + 0j, 3.0 + 0j])
    with pytest.warns(RuntimeWarning, match="branch cut"):
        assert warn_if_near_branch_cut(M, context="test") is True


def test_a_positive_spectrum_does_not_warn():
    M = np.diag([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    with warnings_as_errors():
        assert warn_if_near_branch_cut(M, context="test") is False


def test_an_eigenvalue_off_the_axis_does_not_warn():
    """A negative real part is harmless as long as it is well clear of the
    cut; only near-zero imaginary parts are the hazard."""
    M = np.diag([-1.0 + 2.0j, 3.0 + 0j])
    with warnings_as_errors():
        assert warn_if_near_branch_cut(M, context="test") is False


def warnings_as_errors():
    import warnings

    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("error")
    return _Closing(ctx)


class _Closing:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)
        return False


# --- Shape agnosticism ---


def test_a_two_dimensional_field_keeps_its_shape_and_its_values(hermitian):
    """Krylov projection cares about the vector space, not how it is indexed.

    A field laid out as (Nx, Ny) must give the same numbers as its flattened
    equivalent, and come back in the caller's own shape. This is what lets the
    same integrator serve a 1D and a 2D transverse grid.
    """
    rng = np.random.default_rng(4)
    field = rng.standard_normal((4, 6)) + 1j * rng.standard_normal((4, 6))

    def matvec_2d(x):
        assert x.shape == (4, 6), "matvec must see the caller's shape, not a flat one"
        return (hermitian @ x.reshape(-1)).reshape(4, 6)

    got = krylov_funm(matvec_2d, field, scipy.linalg.expm, max_iter=24, tol=1e-12)
    flat = krylov_funm(
        _matvec(hermitian), field.reshape(-1), scipy.linalg.expm, max_iter=24, tol=1e-12
    )

    assert got.shape == (4, 6)
    np.testing.assert_allclose(got.reshape(-1), flat, atol=1e-12)
    np.testing.assert_allclose(
        got.reshape(-1), scipy.linalg.expm(hermitian) @ field.reshape(-1), atol=1e-9
    )


def test_the_basis_is_flat_whatever_the_input_shape(hermitian):
    """`basis` rows are always flat vectors of length n, so the Arnoldi
    identity can be written as a matrix product regardless of the field
    layout."""
    rng = np.random.default_rng(5)
    field = rng.standard_normal((4, 6)) + 1j * rng.standard_normal((4, 6))

    def matvec_2d(x):
        return (hermitian @ x.reshape(-1)).reshape(4, 6)

    result = arnoldi(matvec_2d, field, max_iter=5)

    assert result.basis.shape == (6, 24)
    assert result.beta == pytest.approx(np.linalg.norm(field))


# --- Backend agnosticism ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_krylov_runs_on_every_backend(hermitian, v, backend_name):
    """The MPS path is single precision, so the tolerance comes from the dtype."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    backend = resolve_backend(backend_name)
    A = backend.xp.asarray(hermitian, dtype=backend.complex, device=backend.device)
    v_backend = backend.xp.asarray(v, dtype=backend.complex, device=backend.device)

    got = to_numpy(
        krylov_funm(_matvec(A), v_backend, scipy.linalg.expm, max_iter=24, tol=1e-10)
    )

    eps = float(np.finfo(got.dtype).eps)
    reference = scipy.linalg.expm(hermitian) @ v
    assert float(np.linalg.norm(got - reference)) < eps**0.5 * float(
        np.linalg.norm(reference)
    )


# --- Truncation at max_iter ---
#
# The propagation operators are exponentiated as exp(i k0 dz H), so the matrix
# handed to Arnoldi is skew-Hermitian and its exponential is unitary. That is
# the case these use: a scaled *Hermitian* matrix underflows expm instead, which
# makes the first-order estimate fire spuriously and tests the wrong thing.


@pytest.fixture
def skew(hermitian):
    """i * H -- skew-Hermitian, so exp of it is unitary, as in a propagation."""
    return 1j * hermitian


@pytest.mark.parametrize("scale", [1.0, 200.0])
@pytest.mark.parametrize(
    "max_iter, expect_warning",
    [
        # Well short of the space: the estimate never comes down.
        (6, True),
        # The full 24 dimensions, where the projection is exact by construction.
        (24, False),
    ],
)
def test_running_out_of_iterations_is_reported(
    skew, v, scale, max_iter, expect_warning
):
    """Truncating at max_iter must not pass for convergence.

    The returned vector has the right shape, a plausible norm and no nans, so a
    silent return is indistinguishable from a converged one -- while being wrong
    by order one rather than by `tol`.
    """
    matvec = _matvec(scale * skew)

    if expect_warning:
        with pytest.warns(RuntimeWarning, match="max_iter"):
            krylov_funm(matvec, v, scipy.linalg.expm, max_iter=max_iter, tol=1e-12)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            krylov_funm(matvec, v, scipy.linalg.expm, max_iter=max_iter, tol=1e-12)


def test_a_truncated_result_is_wrong_by_order_one(skew, v):
    """The reason the warning has to exist: the error is not of order `tol`."""
    A = 200.0 * skew
    reference = scipy.linalg.expm(A) @ v

    with pytest.warns(RuntimeWarning, match="max_iter"):
        truncated = krylov_funm(_matvec(A), v, scipy.linalg.expm, max_iter=6, tol=1e-12)

    relative = np.linalg.norm(truncated - reference) / np.linalg.norm(reference)
    assert relative > 1e-3, "expected an order-one error, not one of order tol"


def test_a_converged_run_agrees_with_the_dense_reference(skew, v):
    """The complement: given enough room, no warning and agreement with dense."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = krylov_funm(_matvec(skew), v, scipy.linalg.expm, max_iter=24, tol=1e-10)

    np.testing.assert_allclose(got, scipy.linalg.expm(skew) @ v, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "stop_given, expected",
    [
        # No stop test: reaching max_iter is exactly what was asked for.
        (False, True),
        # With one, reaching max_iter means it was never satisfied.
        (True, False),
    ],
)
def test_arnoldi_records_why_it_stopped(skew, v, stop_given, expected):
    result = arnoldi(
        _matvec(200.0 * skew),
        v,
        max_iter=6,
        stop=_expm_stop(1e-12) if stop_given else None,
    )
    assert result.converged is expected


def test_breakdown_counts_as_converged(v):
    """A happy breakdown is exact, so it must not be reported as truncation."""
    result = arnoldi(_matvec(np.eye(len(v))), v, max_iter=8, stop=_expm_stop(0.0))
    assert result.breakdown and result.converged
