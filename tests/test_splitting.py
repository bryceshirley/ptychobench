"""Tests for operator-splitting propagation of a separable operator.

The dense path is the accuracy reference throughout: the splittings are
checked against `scipy.linalg.expm` of the full `L + N` matrix, never against
a recorded number.
"""

import warnings

import numpy as np
import pytest
import scipy.linalg

from ptychobench.numerics.apertures import antialias_mask
from ptychobench.grid import SimulationGrid
from ptychobench.numerics.splitting import (
    lie_trotter,
    taylor_exponential,
    warn_if_series_truncated,
)
from ptychobench.utils import resolve_backend, to_numpy


@pytest.fixture
def grid():
    """Small enough for a dense expm, coarse enough to have evanescent modes."""
    return SimulationGrid(N=48, L=20.0, lam=1.0, z_prop=1.0, Nz=1, probe_width=2.0)


@pytest.fixture
def potential(grid):
    """N(x) = sqrt(1 + eps) - 1 for a smooth refractive-index bump."""
    eps = 0.05 * np.exp(-(grid.x**2) / 8.0)
    return np.sqrt(1.0 + eps) - 1.0


@pytest.fixture
def psi0(grid):
    return grid.get_initial_field().astype(complex)


def _dense_reference(grid, potential, total_step):
    """expm(total_step * (L + N)) as a matrix, the accuracy reference."""
    generator = grid.get_angular_spectrum_operator() + np.diag(
        potential.astype(complex)
    )
    return scipy.linalg.expm(total_step * generator)


# --- Exactness where the splitting has nothing to split ---


def test_free_space_is_exact(grid, psi0):
    """With no potential the product collapses to the single kinetic factor,
    which is the analytic angular-spectrum solution."""
    step = 1j * grid.k0 * grid.dz
    got = lie_trotter(psi0, grid.angular_spectrum_symbol, np.zeros(grid.N), step)

    reference = scipy.linalg.expm(step * grid.get_angular_spectrum_operator()) @ psi0
    np.testing.assert_allclose(got, reference, atol=1e-11)


def test_a_zero_kinetic_term_is_exact(grid, psi0, potential):
    """Feit/Fleck is exact by construction when the kinetic operator vanishes:
    the two factors commute trivially because one of them is the identity."""
    step = 1j * grid.k0 * grid.dz
    got = lie_trotter(psi0, np.zeros(grid.N), potential, step)

    reference = np.exp(step * potential) * psi0
    np.testing.assert_allclose(got, reference, atol=1e-13)


# --- Convergence order in the step size ---


def _global_error(grid, psi0, potential, n_steps, total_step):
    """Error after n_steps of size total_step / n_steps, against the dense expm."""
    step = total_step / n_steps
    psi = psi0
    for _ in range(n_steps):
        psi = lie_trotter(psi, grid.angular_spectrum_symbol, potential, step)
    reference = _dense_reference(grid, potential, total_step) @ psi0
    return float(np.linalg.norm(psi - reference))


def test_lie_trotter_is_first_order_in_the_step(grid, psi0, potential):
    """One splitting is on offer, so this is the order of the whole
    ``"split-step"`` integrator and not one option among several.

    Upper bound as well as lower: an ordering that came out at 2 would be a
    symmetric product, which is not what the operators are being told they get.
    """
    total_step = 1j * grid.k0 * grid.z_prop
    counts = np.array([8, 16, 32, 64])
    errors = np.array(
        [_global_error(grid, psi0, potential, n, total_step) for n in counts]
    )
    assert np.all(errors > 0), "an exactly zero error would make the fit meaningless"
    slope, _ = np.polyfit(np.log(1.0 / counts), np.log(errors), 1)

    order = float(slope)
    assert 0.9 < order < 1.2, f"expected global order 1, measured {order:.3f}"


# --- Norm conservation ---


def test_the_norm_is_conserved_for_a_real_potential(grid, psi0, potential):
    """Both symbols are real and the step is imaginary, so every factor has
    unit modulus and the product is unitary. A drift here is a bug, not the
    transform's normalisation -- transforms.fft is unitary by construction."""
    step = 1j * grid.k0 * grid.dz
    psi = psi0
    for _ in range(50):
        psi = lie_trotter(psi, grid.angular_spectrum_symbol, potential, step)

    assert float(np.linalg.norm(psi)) == pytest.approx(
        float(np.linalg.norm(psi0)), rel=1e-12
    )


# --- The cross-term factor ---


@pytest.fixture
def omega(grid):
    """A cross-term-shaped generator: anti-Hermitian, and diagonal in no basis.

    Anti-Hermitian because that is what ``i k0 dz C`` is for a self-adjoint
    cross term, and the unitarity the truncation breaks is only meaningful
    against a generator that has it. Scaled to a spectral radius of 0.3, in the
    regime a real step runs in.
    """
    rng = np.random.default_rng(7)
    a = rng.standard_normal((grid.N, grid.N)) + 1j * rng.standard_normal(
        (grid.N, grid.N)
    )
    skew = a - a.conj().T
    return 0.3 * skew / float(np.max(np.abs(np.linalg.eigvals(skew))))


def test_the_taylor_exponential_matches_a_dense_expm(omega, psi0):
    """The series is the third factor of an H3 or H4 step, so it has to be
    ``exp``, not merely something ``exp``-shaped.

    Checked against `scipy.linalg.expm` rather than against the recurrence's
    own algebra, because the failure this guards against is exactly an algebra
    slip -- Lin and Duda's algorithm as usually transcribed applies the
    factorial twice, which no self-consistent check would catch. At 16 terms
    and a radius of 0.3 the truncation is far below double precision, so any
    residual here is a wrong series and not a short one.
    """
    got = taylor_exponential(lambda v: omega @ v, psi0, order=16)
    np.testing.assert_allclose(got, scipy.linalg.expm(omega) @ psi0, atol=1e-13)


def test_the_taylor_exponential_needs_at_least_one_term(psi0):
    """``order=0`` is the identity, which is never what a caller meant."""
    with pytest.raises(ValueError, match="at least 1"):
        taylor_exponential(lambda v: v, psi0, order=0)


def test_truncating_the_series_costs_unitarity(omega, psi0):
    """The exact exponential of an anti-Hermitian generator is unitary and no
    truncation of it is. Asserted rather than left in a docstring because it is
    the reason ``_step_split`` warns, and the reason a split-step H4 run does
    not conserve norm the way an H2 one does."""
    drift = abs(
        float(np.linalg.norm(taylor_exponential(lambda v: omega @ v, psi0, order=2)))
        / float(np.linalg.norm(psi0))
        - 1.0
    )
    assert 0.0 < drift < 1e-2


@pytest.mark.parametrize(
    ("order", "expected"), [(2, True), (16, False)], ids=["short", "ample"]
)
def test_the_truncation_warning_tracks_the_retained_order(omega, psi0, order, expected):
    """One ratio, two verdicts: whether the series is long enough is a question
    about ``order``, not about ``Omega`` alone, so the same generator has to
    warn at 2 terms and stay quiet at 16."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warned = warn_if_series_truncated(
            omega @ psi0, psi0, order=order, context="test"
        )

    assert warned is expected
    assert bool(caught) is expected


def test_the_truncation_warning_tolerates_a_zero_field(psi0):
    """The ratio is undefined at ``||psi|| == 0``, and a zero field is a
    legitimate thing to propagate. Returning False beats dividing by zero."""
    zero = np.zeros_like(psi0)
    assert warn_if_series_truncated(zero, zero, order=1, context="test") is False


# --- Band limiting ---


def test_the_aperture_is_composed_with_the_kinetic_factor(grid, psi0, potential):
    """The documented action is P K V, with the projection folded into the
    same transform pair as the kinetic factor rather than into its exponent."""
    step = 1j * grid.k0 * grid.dz
    mask = antialias_mask(grid.kx)

    got = lie_trotter(
        psi0, grid.angular_spectrum_symbol, potential, step, aperture=mask
    )

    multiplier = mask * np.exp(step * grid.angular_spectrum_symbol)
    expected = np.fft.ifft(multiplier * np.fft.fft(np.exp(step * potential) * psi0))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_the_aperture_does_not_commute_with_the_potential(grid, psi0):
    """Stated in the module docstring, asserted here rather than assumed.

    This is why band limiting changes the operator being integrated and not
    merely how it is evaluated. The potential here is a sharp slab: a smooth,
    weak bump has content nowhere near the band edge and the two orderings
    then agree to about 1e-11, which would make this test pass for the wrong
    reason.
    """
    mask = antialias_mask(grid.kx)
    slab = np.where(np.abs(grid.x) < 3.0, 0.2, 0.0)

    def project(f):
        return np.fft.ifft(mask * np.fft.fft(f))

    def apply_potential(f):
        return np.exp(slab) * f

    difference = project(apply_potential(psi0)) - apply_potential(project(psi0))
    relative = float(np.linalg.norm(difference)) / float(np.linalg.norm(psi0))
    assert relative > 1e-3, f"commutator is only {relative:.2e} of the field"


def test_a_lie_trotter_step_leaves_the_field_band_limited(grid, psi0, potential):
    """P K V ends on the projection, so the step's output is band limited."""
    step = 1j * grid.k0 * grid.dz
    mask = antialias_mask(grid.kx)

    psi = lie_trotter(
        psi0, grid.angular_spectrum_symbol, potential, step, aperture=mask
    )

    np.testing.assert_allclose(np.fft.fft(psi)[mask == 0.0], 0.0, atol=1e-14)


def test_a_trailing_real_space_kick_refills_the_stopband(grid, psi0, potential):
    """The other face of the same non-commutation.

    The test above is a property of the *ordering*, not of splitting, and the
    module docstring says so; this is what makes that qualification honest. A
    product ending on a real-space factor -- the symmetric Strang form that
    used to live here, or any caller who kicks after stepping -- repopulates
    the stopband the projection just cleared. So a reader must not carry "a
    band-limited step returns a band-limited field" over to a rearrangement of
    the factors.
    """
    step = 1j * grid.k0 * grid.dz
    mask = antialias_mask(grid.kx)

    band_limited = lie_trotter(
        psi0, grid.angular_spectrum_symbol, potential, step, aperture=mask
    )
    kicked = np.exp(step * potential) * band_limited

    assert float(np.linalg.norm(np.fft.fft(kicked)[mask == 0.0])) > 0.0


def test_a_binary_aperture_is_idempotent_across_steps(grid, psi0):
    """P P = P for edge="binary", so applying it once per step is the same
    projection every step rather than a slowly narrowing one.

    With no potential the factors all commute, so two steps of size h must
    equal one application of P exp(2h L) exactly -- and would not if the
    projection compounded.
    """
    step = 1j * grid.k0 * grid.dz
    mask = antialias_mask(grid.kx)
    zero = np.zeros(grid.N)

    once = lie_trotter(psi0, grid.angular_spectrum_symbol, zero, step, aperture=mask)
    twice = lie_trotter(once, grid.angular_spectrum_symbol, zero, step, aperture=mask)

    combined = np.fft.ifft(
        mask * np.exp(2.0 * step * grid.angular_spectrum_symbol) * np.fft.fft(psi0)
    )
    np.testing.assert_allclose(twice, combined, atol=1e-13)


# --- Backend agnosticism ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_splitting_runs_on_every_backend(grid, psi0, potential, backend_name):
    """The MPS path is single precision, so the tolerance comes from the dtype."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    backend = resolve_backend(backend_name)

    def as_backend(a, dt):
        return backend.xp.asarray(a, dtype=dt, device=backend.device)

    step = 1j * grid.k0 * grid.dz
    got = to_numpy(
        lie_trotter(
            as_backend(psi0, backend.complex),
            as_backend(grid.angular_spectrum_symbol, backend.real),
            as_backend(potential, backend.real),
            step,
        )
    )

    reference = to_numpy(
        lie_trotter(psi0, grid.angular_spectrum_symbol, potential, step)
    )
    eps = float(np.finfo(got.dtype).eps)
    assert float(np.linalg.norm(got - reference)) < eps**0.5 * float(
        np.linalg.norm(reference)
    )
