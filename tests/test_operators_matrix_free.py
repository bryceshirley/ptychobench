"""Tests for the matrix-free propagation paths on the forward operators.

The dense path is the accuracy reference throughout. Every matrix-free action
is checked against the operator matrix `construct_operator` builds, and every
matrix-free step against the dense `expm` step -- never against a recorded
number.
"""

from typing import cast

import numpy as np
import pytest

from ptychobench.numerics.apertures import antialias_mask
from ptychobench.grid import SimulationGrid
from ptychobench.operators import (
    GroundTruthOperator,
    FeitFleckOperator,
    Integrator,
    LinDudaOperator,
    ParaxialOperator,
    YevickThomsonOperator,
)
from ptychobench.numerics.transforms import fourier_multiply
from ptychobench.utils import to_numpy

#: Everything except GroundTruthOperator, whose operator is a matrix square root and
#: so has no cheap elementwise action; it gets its own tests below.
APPROXIMATIONS = [
    ParaxialOperator,
    FeitFleckOperator,
    YevickThomsonOperator,
    LinDudaOperator,
]

#: The operators with a split-step form -- which is all four approximations.
#: H1 and H2 separate outright into a Fourier-diagonal and a real-space-diagonal
#: term; the cross terms in H3 and H4 are products of the two, which no
#: two-factor product can represent, so those ride in a third factor evaluated
#: as a truncated series. Only the reference is missing, having no such
#: decomposition at all.
SPLIT_STEP_CAPABLE = APPROXIMATIONS


@pytest.fixture
def grid():
    return SimulationGrid(N=64, L=20.0, lam=1.0, z_prop=1.0, Nz=100, probe_width=2.0)


@pytest.fixture
def eps(grid):
    """A refractive-index perturbation with structure across the band.

    Deliberately non-negative. The evanescent clip sends ``1 + mu`` to exactly
    zero at the band edge, so a permittivity that dips below zero makes
    ``A = 1 + mu + eps`` indefinite and puts ``sqrt(A)`` on its branch cut,
    where the dense and Krylov paths disagree by around 1e-8 no matter how
    large the subspace. That is a real effect and it has its own test below;
    letting it leak into every comparison would only obscure them.
    """
    return 0.08 * np.exp(-(grid.x**2) / 6.0) * (1.0 + 0.5 * np.cos(2.0 * grid.x))


@pytest.fixture
def psi0(grid):
    return grid.get_initial_field().astype(complex)


# --- The matrix-free action reproduces the dense operator ---


@pytest.mark.parametrize("OpClass", APPROXIMATIONS)
def test_apply_reproduces_the_dense_operator(grid, eps, psi0, OpClass):
    """H psi computed matrix-free must equal the dense H times psi.

    This is the test that pins the cross-term algebra: a wrong sign or a
    transposed factor in `apply` shows up here immediately and unambiguously,
    with no propagation or matrix function in the way to mask it.
    """
    op = OpClass(grid)

    got = op.apply(eps, psi0)
    reference = op.construct_operator(np.diag(eps.astype(complex))) @ psi0

    np.testing.assert_allclose(got, reference, atol=1e-11)


@pytest.mark.parametrize("OpClass", APPROXIMATIONS)
def test_apply_is_linear(grid, eps, psi0, OpClass):
    op = OpClass(grid)
    other = np.roll(psi0, 7)

    combined = op.apply(eps, 2.0 * psi0 - 3.0j * other)
    separate = 2.0 * op.apply(eps, psi0) - 3.0j * op.apply(eps, other)

    np.testing.assert_allclose(combined, separate, atol=1e-12)


# --- Krylov agrees with the dense step ---


@pytest.mark.parametrize("OpClass", APPROXIMATIONS)
def test_the_krylov_step_matches_the_direct_step(grid, eps, psi0, OpClass):
    direct = OpClass(grid, integrator="direct")
    krylov = OpClass(grid, integrator="krylov")

    E = np.diag(eps.astype(complex))
    np.testing.assert_allclose(krylov.step(E, psi0), direct.step(E, psi0), atol=1e-9)


def test_the_exact_operator_krylov_step_matches_the_direct_step(grid, eps, psi0):
    """The exact operator needs a composed function of the projection.

    f(z) = exp(dz_factor * (sqrt(z) - 1)) is applied to the projection of
    A = 1 + mu + eps, so one Arnoldi run covers both the square root and the
    exponential instead of nesting a Krylov solve inside another.

    The tolerance is looser than for the other operators, and not because the
    subspace is too small: the discrepancy sits at ~2e-9 even with a *complete*
    subspace (max_iter = N). The evanescent clip pins 1 + mu to exactly zero at
    the band edge, which is the branch point of the square root, and sqrtm's
    condition number grows like 1/sqrt(lambda) there. Both paths are converged;
    the quantity itself is only determined to about that level.
    """
    direct = GroundTruthOperator(grid, integrator="direct")
    krylov = GroundTruthOperator(grid, integrator="krylov")

    E = np.diag(eps.astype(complex))
    reference = direct.step(E, psi0)
    error = float(np.linalg.norm(krylov.step(E, psi0) - reference))
    assert error / float(np.linalg.norm(reference)) < 1e-7


@pytest.mark.parametrize("OpClass", APPROXIMATIONS + [GroundTruthOperator])
def test_the_krylov_step_accepts_the_permittivity_as_a_vector(grid, eps, psi0, OpClass):
    """The matrix-free path must not require an N x N matrix to be built just
    to carry N numbers down the diagonal."""
    op = OpClass(grid, integrator="krylov")

    np.testing.assert_allclose(
        op.step(eps, psi0), op.step(np.diag(eps.astype(complex)), psi0), atol=1e-12
    )


# --- Split-step ---


@pytest.mark.parametrize("OpClass", SPLIT_STEP_CAPABLE)
def test_the_split_step_converges_to_the_direct_step(grid, eps, psi0, OpClass):
    """One step, so this is the *local* error against the dense exponential of
    the same generator: order 2 for Lie-Trotter, one more than its global order.

    The bound is two-sided on purpose. This measured 3 while ``_step_split``
    used the symmetric Strang product, and the old assertion was a bare
    ``> 1.8`` that a second-order local error would also have satisfied --
    so it could not have told the two products apart. Dropping to one ordering
    traded that order away deliberately (see :mod:`ptychobench.numerics.splitting`), and
    a slope back near 3 would mean the splitting had quietly become symmetric
    again.

    For H3 and H4 the same bounds do a second job. The three-factor product is
    first order for the same reason the two-factor one is, so the slope is
    still 2 -- but only if the third factor is actually there. Dropping
    ``exp(Omega)`` would leave an error of ``h C psi`` at leading order and the
    slope would fall to 1, well under the lower bound. The series truncation
    itself contributes at ``h^(M+1)``, far below the floor, so it does not
    show up here; ``test_the_series_warns_when_the_step_is_too_large`` covers
    the regime where it does.
    """
    errors = []
    for nz in (25, 50, 100, 200):
        fine = SimulationGrid(
            N=grid.N, L=grid.L, lam=grid.lam, z_prop=grid.z_prop, Nz=nz
        )
        direct = OpClass(fine, integrator="direct")
        split = OpClass(fine, integrator="split-step")

        E = np.diag(eps.astype(complex))
        errors.append(float(np.linalg.norm(split.step(E, psi0) - direct.step(E, psi0))))

    errors_arr = np.array(errors)
    slopes = np.log2(errors_arr[:-1] / errors_arr[1:])
    assert np.all(slopes > 1.8), f"expected local order 2, measured {slopes}"
    assert np.all(slopes < 2.5), f"that is not a Lie-Trotter slope: {slopes}"


def test_the_exact_operator_rejects_split_step(grid):
    """``sqrt(1 + mu + eps)`` is not a sum of diagonal pieces in any basis and
    has no cross term to peel off either, so there is nothing to split.
    Refusing is the honest answer; silently dropping terms would not be, and
    the reference is the one operator that cannot afford to.
    """
    with pytest.raises(NotImplementedError, match="separate"):
        GroundTruthOperator(grid, integrator="split-step")


@pytest.mark.parametrize("OpClass", [YevickThomsonOperator, LinDudaOperator])
def test_the_cross_term_is_the_difference_from_feit_fleck(grid, eps, psi0, OpClass):
    """``split_step_parts`` on H3 and H4 returns Feit/Fleck's two factors, and
    the whole of the difference lives in ``cross_term_action``.

    Pinned against ``apply`` -- which is itself pinned against the dense matrix
    -- so the split and Krylov paths cannot end up with different ideas of what
    the cross term is. A sign slip in one of them would show up here rather
    than as a convergence rate that is merely a little worse than expected.
    """
    op = OpClass(grid, integrator="split-step")
    h2 = FeitFleckOperator(grid, integrator="krylov")

    np.testing.assert_allclose(
        op.apply(eps, psi0),
        h2.apply(eps, psi0) + op.cross_term_action(eps, psi0),
        atol=1e-14,
    )
    kinetic, potential = op.split_step_parts(eps)
    np.testing.assert_array_equal(kinetic, h2.split_step_parts(eps)[0])
    np.testing.assert_array_equal(potential, h2.split_step_parts(eps)[1])


@pytest.fixture
def coarse_grid(grid):
    """The same grid taking one enormous step instead of a hundred small ones.

    ``||Omega||`` is 2.7e-05 on the `grid` fixture, so the shipped four-term
    series is exact to 1e-25 there and no amount of propagating will provoke
    it. It is proportional to ``dz``, so a step 2e4 times longer puts it at
    0.54 -- the regime the warning exists for, reached by making the step too
    large rather than by shortening the series, which is the mistake a caller
    would actually make.
    """
    return SimulationGrid(
        N=grid.N, L=grid.L, lam=grid.lam, z_prop=200.0, Nz=1, probe_width=2.0
    )


@pytest.mark.parametrize("OpClass", [YevickThomsonOperator, LinDudaOperator])
def test_the_series_warns_when_the_step_is_too_large(coarse_grid, eps, psi0, OpClass):
    """The truncated series is the one part of the split step with a horizon,
    so a step past it must say so rather than return a quietly wrong field."""
    op = OpClass(coarse_grid, integrator="split-step")
    with pytest.warns(RuntimeWarning, match="truncated"):
        op.step(eps, psi0)


def test_the_default_series_length_is_silent_on_this_package_s_own_grids(
    grid, eps, psi0, recwarn
):
    """The counterpart to the test above, and the one that keeps the default
    honest: four terms must be ample in the regime the benchmark actually runs
    in, or every run would carry a warning nobody could act on."""
    LinDudaOperator(grid, integrator="split-step").step(eps, psi0)

    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_the_truncation_warning_is_latched(coarse_grid, eps, psi0, recwarn):
    """Once per propagation, not once per step. The ratio it reports is set by
    the grid and the step, so it does not change between steps, and a
    thousand-step run would otherwise emit a thousand copies -- and each check
    costs a whole extra application of ``Omega``."""
    op = LinDudaOperator(coarse_grid, integrator="split-step")
    field = psi0
    for _ in range(5):
        field = op.step(eps, field)

    assert len([w for w in recwarn if issubclass(w.category, RuntimeWarning)]) == 1


# --- Dispatch ---


def test_an_unknown_integrator_is_rejected_at_construction(grid):
    """Fail where the mistake was made, not on the first propagation step.

    The `cast` is the point of the test, not a workaround: `Integrator` is a
    Literal, so a type-checked caller cannot reach this guard, and the guard
    exists for the ones that are not -- a name read from a config file or a
    command line. Casting is how the test gets to stand where they stand.
    """
    with pytest.raises(ValueError, match="unknown integrator"):
        FeitFleckOperator(grid, integrator=cast("Integrator", "lanczos"))


@pytest.mark.parametrize("OpClass", APPROXIMATIONS + [GroundTruthOperator])
def test_the_direct_path_is_unchanged_by_the_dispatch(grid, eps, psi0, OpClass):
    """The dense path is the accuracy reference and must stay bit-for-bit what
    it was: expm of the constructed operator, applied to the field."""
    from scipy.linalg import expm

    op = OpClass(grid)
    E = np.diag(eps.astype(complex))

    expected = expm(op.dz_factor * op.construct_operator(E)) @ psi0
    np.testing.assert_array_equal(op.step(E, psi0), expected)


# --- The branch cut of the exact operator's square root ---


#: A refractive-index dip deep enough to drive the *projection* indefinite.
#:
#: The guard inspects the small Hessenberg matrix that sqrtm is actually
#: applied to, not the full operator, and that is the right thing to inspect:
#: Ritz values interlace the spectrum, so a weakly indefinite operator can be
#: projected onto a subspace whose own spectrum is entirely positive. A -0.02
#: dip makes A indefinite (25 negative eigenvalues) yet the tolerance is met at
#: m = 8 with every Ritz value positive, and no warning is due. -0.5 is deep
#: enough that the negative part is captured from the first few vectors.
_INDEFINITE_DIP = -0.5


def test_a_negative_permittivity_warns_about_the_branch_cut(grid, psi0):
    """sqrt(1 + mu + eps) is ill-conditioned when the argument goes indefinite.

    The principal root jumps across the negative real axis, so its value there
    is not determined by the data. Worth a warning rather than a silently
    degraded number.
    """
    dip = _INDEFINITE_DIP * np.exp(-(grid.x**2) / 6.0)
    op = GroundTruthOperator(grid, integrator="krylov")

    with pytest.warns(RuntimeWarning, match="branch cut"):
        op.step(dip, psi0)


def test_the_branch_cut_warning_is_raised_once_per_operator(grid, psi0, recwarn):
    """One warning describes the configuration.

    A propagation loop evaluates the square root once per Arnoldi iteration of
    every step -- thousands of times -- and must not say so thousands of times.
    The warning also carries the offending eigenvalue, which differs between
    calls, so Python's own duplicate suppression would not collapse them.
    """
    dip = _INDEFINITE_DIP * np.exp(-(grid.x**2) / 6.0)
    op = GroundTruthOperator(grid, integrator="krylov")

    for _ in range(5):
        op.step(dip, psi0)

    branch_cut = [w for w in recwarn if "branch cut" in str(w.message)]
    assert len(branch_cut) == 1, f"got {len(branch_cut)} warnings"


# --- Self-adjointness, and where it stops holding ---


@pytest.fixture
def probe_pair(grid):
    """Two unrelated complex fields, for testing <u, A v> == <A u, v>."""
    rng = np.random.default_rng(11)
    shape = (grid.N,)
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
    )


@pytest.mark.parametrize("OpClass", APPROXIMATIONS)
def test_the_generator_is_self_adjoint_for_a_real_permittivity(
    grid, eps, probe_pair, OpClass
):
    """<u, H v> == <H u, v>, which is what makes the step unitary.

    This is the structural reason the cross terms in H3 and H4 are
    symmetrised. Both -(eps mu + mu eps)/8 and -eps mu / 4 have the same scalar
    Taylor expansion, but only the first is self-adjoint; the second would
    still pass a convergence-order test and quietly lose norm conservation.
    """
    op = OpClass(grid)
    u, v = probe_pair

    left = complex(np.vdot(u, op.apply(eps, v)))
    right = complex(np.vdot(op.apply(eps, u), v))

    assert abs(left - right) < 1e-12 * abs(left)


def test_the_band_limiting_projection_is_self_adjoint(grid, probe_pair):
    """P is a real, even Fourier multiplier, so it is its own adjoint."""
    u, v = probe_pair
    mask = antialias_mask(grid.kx)

    left = complex(np.vdot(u, fourier_multiply(v, mask)))
    right = complex(np.vdot(fourier_multiply(u, mask), v))

    assert abs(left - right) < 1e-12 * abs(left)


def test_the_projected_generator_is_not_self_adjoint(grid, eps, probe_pair):
    """P and H are each self-adjoint; their product is not.

    ``(P H)* = H P``, which equals ``P H`` only if the two commute -- and they
    do not, because P is diagonal in Fourier space and the potential is
    diagonal in real space. So composing the band limit onto a step changes the
    operator being integrated, and the unitarity argument that norm
    conservation rests on no longer applies to it. Stated in the splitting
    module's docstring; asserted here on the operator it actually affects.
    """
    op = FeitFleckOperator(grid)
    u, v = probe_pair
    mask = antialias_mask(grid.kx)

    left = complex(np.vdot(u, fourier_multiply(op.apply(eps, v), mask)))
    right = complex(np.vdot(fourier_multiply(op.apply(eps, u), mask), v))

    assert abs(left - right) > 1e-6 * abs(left), (
        "P H came out self-adjoint, so this configuration is not exercising "
        "the non-commutation the test exists to demonstrate"
    )


# --- Split-step against Krylov ---


def test_the_split_step_and_krylov_paths_converge_to_each_other(grid, eps, psi0):
    """The two matrix-free paths are independent of one another.

    Both are checked against the dense reference elsewhere, but that leaves
    room for a shared misreading of the generator. Krylov is exact to 1e-9 at
    any step, so this measures the splitting error alone -- one step of it, so
    Lie-Trotter's local order 2 rather than its global order 1.
    """
    errors = []
    for nz in (25, 50, 100, 200):
        fine = SimulationGrid(
            N=grid.N, L=grid.L, lam=grid.lam, z_prop=grid.z_prop, Nz=nz
        )
        krylov = FeitFleckOperator(fine, integrator="krylov")
        split = FeitFleckOperator(fine, integrator="split-step")

        errors.append(
            float(np.linalg.norm(split.step(eps, psi0) - krylov.step(eps, psi0)))
        )

    slopes = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(slopes > 1.8), f"expected local order 2, measured {slopes}"


# --- Norm conservation ---


@pytest.mark.parametrize("integrator", ["direct", "krylov"])
def test_the_norm_is_conserved_for_a_real_permittivity(grid, eps, psi0, integrator):
    """A real permittivity makes the generator self-adjoint and the step
    unitary, so the field keeps its 2-norm."""
    op = FeitFleckOperator(grid, integrator=integrator)
    E = np.diag(eps.astype(complex))

    psi = psi0
    for _ in range(20):
        psi = op.step(E, psi)

    assert float(np.linalg.norm(psi)) == pytest.approx(
        float(np.linalg.norm(psi0)), rel=1e-8
    )


# =========================================================
# Backends
# =========================================================
def _propagate(OpClass, integrator: str, backend: str, steps: int = 4):
    """Four steps of one operator, end to end from the grid, on one backend."""
    grid = SimulationGrid(
        N=64, L=20.0, lam=1.0, z_prop=1.0, Nz=100, probe_width=2.0, backend=backend
    )
    xp = grid.xp
    eps = 0.08 * xp.exp(-(grid.x**2) / 6.0) * (1.0 + 0.5 * xp.cos(2.0 * grid.x))
    psi = grid.get_initial_field()

    operator = OpClass(grid, integrator=cast(Integrator, integrator))
    for _ in range(steps):
        psi = operator.step(xp.astype(eps, grid.complex_dtype), psi)
    return psi


@pytest.mark.parametrize("OpClass", APPROXIMATIONS + [GroundTruthOperator])
@pytest.mark.parametrize("integrator", ["direct", "krylov", "split-step"])
def test_every_integrator_runs_on_a_torch_grid(OpClass, integrator):
    """The GPU exists for the matrix-free paths, but "direct" is the *default*,
    so it has to work on a device grid too -- on the host, since it hands a
    dense matrix to scipy, but without the caller having to know that.

    Torch on Apple Silicon is float32 -- Metal has no FP64 -- so this is a
    single-precision agreement check against the float64 reference, not a
    bit-for-bit one. It runs end to end from the grid, which is the only way to
    catch a symbol that is finite in double and NaN in single.
    """
    pytest.importorskip("torch", reason="the torch backend is optional")
    if integrator == "split-step" and not OpClass.supports_split_step():
        pytest.skip("this operator has no split-step form")

    np.testing.assert_allclose(
        to_numpy(_propagate(OpClass, integrator, "torch")),
        to_numpy(_propagate(OpClass, integrator, "numpy")),
        atol=1e-5,
    )


@pytest.mark.parametrize("integrator", ["direct", "krylov", "split-step"])
def test_a_step_returns_a_field_in_the_grids_backend(integrator):
    """Whichever integrator advanced it, the field comes back where it started.

    "direct" is the one that could plausibly not: it computes on the host in
    NumPy. Returning that host array unchanged would mean the *default*
    integrator silently drops a torch run onto the CPU after one step, and
    crashes outright on MPS, where numpy conversion is refused.
    """
    torch = pytest.importorskip("torch", reason="the torch backend is optional")

    psi = _propagate(ParaxialOperator, integrator, "torch", steps=1)

    assert isinstance(psi, torch.Tensor)


@pytest.mark.parametrize("OpClass", SPLIT_STEP_CAPABLE)
def test_the_split_step_carries_a_gradient_back_to_the_permittivity(OpClass):
    """The one thing `"split-step"` can do that the other two integrators
    cannot, and the stated reason H3 and H4 were given a third factor at all.

    Checked against a central difference rather than merely for a non-None
    `.grad`, because the failure worth catching is a path that detaches *part*
    of the graph -- a `float()` on an intermediate, or a host round trip -- and
    still produces a plausible-looking gradient from what is left. Only the
    numbers distinguish that from the real thing.

    Deliberately float64 on the CPU: the finite difference needs precision the
    MPS path does not have. Whether the *kernels* run on a device is a separate
    question, covered above.
    """
    torch = pytest.importorskip("torch", reason="the torch backend is optional")

    grid = SimulationGrid(
        N=32,
        L=10.0,
        lam=1.0,
        z_prop=0.5,
        Nz=5,
        probe_width=2.0,
        backend="torch",
        device="cpu",
    )
    baseline = 0.08 * np.exp(-(to_numpy(grid.x) ** 2) / 6.0)
    psi0 = torch.as_tensor(to_numpy(grid.get_initial_field()), dtype=torch.complex128)
    # A weighted intensity, not the total: the propagator is near-unitary, so
    # the total is conserved and its gradient is zero to round-off, which every
    # broken implementation would also reproduce.
    weight = torch.as_tensor(np.linspace(0.0, 1.0, grid.N))

    def loss(permittivity):
        op = OpClass(grid, integrator="split-step")
        field = psi0
        for _ in range(5):
            field = op.step(permittivity, field)
        return (weight * torch.abs(field) ** 2).sum()

    eps = torch.tensor(baseline, requires_grad=True)
    loss(eps).backward()
    assert eps.grad is not None

    # rel=1e-4 is the central difference's own accuracy at this h, not slack in
    # the gradient: a detached path would be wrong by a factor, not by 1e-4.
    h = 1e-6
    for i in (5, 10, 16, 20):
        shifted = np.tile(baseline, (2, 1))
        shifted[0, i] += h
        shifted[1, i] -= h
        with torch.no_grad():
            up, down = (loss(torch.as_tensor(row)) for row in shifted)
        assert float(eps.grad[i]) == pytest.approx(
            float((up - down) / (2 * h)), rel=1e-4
        )
