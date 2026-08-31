"""Convergence order of the operator hierarchy in the perturbation.

The operators H1..H4 are truncations of

    H = sqrt(1 + mu + eps) - 1

in powers of the two perturbations together, so their order of accuracy is an
order in ``eps`` and ``mu`` -- *not* in the step size ``dz``, which does not
appear in the generator at all. Advancing a field and fitting against ``dz``
would measure the exponential's own discretisation and say nothing about which
approximation is which.

The scaling knob is a single ``s`` applied to both perturbations at once:

* ``eps -> s * eps`` directly.
* ``mu = -kx^2 / k0^2 -> s * mu`` by taking ``k0 -> k0 / sqrt(s)``, that is
  ``lam -> lam * sqrt(s)``.

So ``s -> 0`` is the joint weak-scattering and small-angle limit, which is the
regime every one of these expansions is derived in. Shrinking ``k0`` this way
also widens the propagating band, so the evanescent clip stops biting and the
square root stays away from its branch point -- the comparison is then measuring
truncation and nothing else.

Expanding both sides in the combined perturbation gives

    H = (mu + eps)/2 - (mu + eps)^2/8 + ...

so an operator that matches through first order has error ``O(s^2)`` and one
that matches through second order has error ``O(s^3)``. Against the "order of
accuracy" column in docs/technical.md:

    H1 Paraxial        first order    error O(s^2)
    H2 Feit/Fleck      first order    error O(s^2)
    H3 Yevick/Thomson  second order   error O(s^3)
    H4 Lin/Duda        second order   error O(s^3)

The comparison is between dense generators rather than between propagated
fields, so no matrix function sits between the algebra and the measurement.
"""

import numpy as np
import pytest

from ptychobench.grid import SimulationGrid
from ptychobench.operators import (
    GroundTruthOperator,
    FeitFleckOperator,
    LinDudaOperator,
    ParaxialOperator,
    YevickThomsonOperator,
)

#: Geometry held fixed while the perturbation is scaled.
N = 48
L = 20.0
LAM0 = 1.0

#: Halving each time, so successive ratios are a clean factor of two.
SCALES = np.array([0.4, 0.2, 0.1, 0.05, 0.025])

FIRST_ORDER = [ParaxialOperator, FeitFleckOperator]
SECOND_ORDER = [YevickThomsonOperator, LinDudaOperator]


def _generator_error(OpClass, scale):
    """Frobenius distance from the exact generator at perturbation ``scale``."""
    grid = SimulationGrid(N=N, L=L, lam=LAM0 * np.sqrt(scale), z_prop=1.0, Nz=1)
    E = np.diag((scale * 0.5 * np.exp(-(grid.x**2) / 8.0)).astype(complex))

    approximate = OpClass(grid).construct_operator(E)
    exact = GroundTruthOperator(grid).construct_operator(E)
    return float(np.linalg.norm(approximate - exact))


def _asymptotic_order(OpClass):
    """The order measured on the smallest pair of scales.

    Deliberately the finest pair rather than a fit over all of them. The
    pre-asymptotic error of a *wrong* operator can look like the right order at
    s = 0.4 -- an unsymmetrised cross term measures 3.29 there before settling
    to 2 -- so a fit across the whole range would blur exactly the distinction
    this module exists to make.
    """
    coarse, fine = SCALES[-2], SCALES[-1]
    errors = (_generator_error(OpClass, coarse), _generator_error(OpClass, fine))
    return float(np.log(errors[0] / errors[1]) / np.log(coarse / fine))


@pytest.mark.parametrize("OpClass", FIRST_ORDER)
def test_a_first_order_operator_has_a_second_order_error(OpClass):
    order = _asymptotic_order(OpClass)
    assert 1.8 < order < 2.5, f"expected error order 2, measured {order:.3f}"


@pytest.mark.parametrize("OpClass", SECOND_ORDER)
def test_a_second_order_operator_has_a_third_order_error(OpClass):
    """The cross term is what buys the extra order, so this is the test that
    pins its coefficient -- not just its presence."""
    order = _asymptotic_order(OpClass)
    assert 2.8 < order < 3.3, f"expected error order 3, measured {order:.3f}"


def test_the_error_shrinks_monotonically_with_the_perturbation():
    """A non-monotone sequence would make the two-point slope meaningless."""
    for OpClass in FIRST_ORDER + SECOND_ORDER:
        errors = [_generator_error(OpClass, s) for s in SCALES]
        assert all(b < a for a, b in zip(errors, errors[1:])), (
            f"{OpClass.__name__} errors are not decreasing: {errors}"
        )


def test_the_paraxial_operator_is_wrong_even_in_free_space():
    """H1 truncates the kinetic term too, so its error does not vanish with the
    permittivity: at eps = 0 it gives mu/2 where the exact answer is
    sqrt(1 + mu) - 1. This is what separates it from H2, which the order test
    alone cannot see -- both are first order.

    The tolerance on the H2 side is 1e-7 relative rather than round-off, and
    that is a property of the *reference*, not of H2. At the full wavelength the
    evanescent clip pins 1 + mu to exactly zero above the band edge, so
    ``GroundTruthOperator`` hands ``scipy.linalg.sqrtm`` a singular matrix. Its
    condition number is unbounded at the branch point, and the dense exact
    generator is itself only good to about 1e-8 there. H2 gets the same quantity
    from the analytic symbol and is the more accurate of the two.
    """
    grid = SimulationGrid(N=N, L=L, lam=LAM0, z_prop=1.0, Nz=1)
    vacuum = np.zeros((grid.N, grid.N), dtype=complex)

    exact = GroundTruthOperator(grid).construct_operator(vacuum)
    paraxial = ParaxialOperator(grid).construct_operator(vacuum)
    feit_fleck = FeitFleckOperator(grid).construct_operator(vacuum)

    reference_norm = float(np.linalg.norm(exact))
    assert float(np.linalg.norm(paraxial - exact)) / reference_norm > 0.1
    assert float(np.linalg.norm(feit_fleck - exact)) / reference_norm < 1e-7


# --- The order test has to be able to fail ---


class _FlippedSignYevick(YevickThomsonOperator):
    """H3 with the cross term added instead of subtracted."""

    def construct_operator(self, E):
        N_op = np.sqrt(1 + E) - 1.0
        eps_vec = np.diag(E)
        cross = (1 / 8) * ((self.mu_op * eps_vec) + (self.mu_op * eps_vec[:, None]))
        return self.L_op + N_op + cross


class _UnsymmetrisedYevick(YevickThomsonOperator):
    """H3 with ``-eps mu / 4`` in place of ``-(eps mu + mu eps) / 8``.

    Same trace and the same scalar Taylor expansion, so it is an easy mistake
    to make and an easy one to miss; it also destroys self-adjointness and
    hence norm conservation.
    """

    def construct_operator(self, E):
        N_op = np.sqrt(1 + E) - 1.0
        eps_vec = np.diag(E)
        return self.L_op + N_op - (1 / 4) * (self.mu_op * eps_vec)


class _DroppedCrossLinDuda(LinDudaOperator):
    """H4 with no cross term at all, which makes it H2."""

    def construct_operator(self, E):
        eps_vec = np.diag(E)
        return self.L_op + np.diag(np.sqrt(1 + eps_vec) - 1.0)


@pytest.mark.parametrize(
    "OpClass", [_FlippedSignYevick, _UnsymmetrisedYevick, _DroppedCrossLinDuda]
)
def test_a_broken_cross_term_loses_the_extra_order(OpClass):
    """Guard on the guard.

    A test that measures an order is only worth having if a wrong operator
    fails it. Each of these three is a plausible slip -- a sign, a missing
    symmetrisation, an omission -- and each must drop back to order 2.
    """
    order = _asymptotic_order(OpClass)
    assert order < 2.5, f"a broken cross term still measured order {order:.3f}"
