"""Tests for the built-in 1D band-limiting aperture."""

import numpy as np
import pytest

from ptychobench.numerics.apertures import antialias_mask
from ptychobench.grid import SimulationGrid
from ptychobench.utils import resolve_backend, to_numpy


@pytest.fixture
def kx() -> np.ndarray:
    """Fourier coordinates from a real grid, so the tests use the real layout."""
    return SimulationGrid(N=64).kx


def _nyquist(kx: np.ndarray) -> float:
    return float(np.max(np.abs(kx)))


# --- The projection property, which is the whole point of `edge` ---


def test_binary_mask_is_a_projection(kx):
    """P^2 == P exactly, so composing the aperture twice changes nothing."""
    mask = antialias_mask(kx)
    np.testing.assert_array_equal(mask * mask, mask)


def test_tapered_mask_is_not_a_projection(kx):
    """A raised-cosine roll-off is a filter, not a projection: P^2 != P.

    This is not a defect to be fixed -- it is the reason `edge` is an explicit
    choice rather than a hidden default.
    """
    mask = antialias_mask(kx, edge="tapered", taper=0.1)
    idempotency_defect = float(np.max(np.abs(mask * mask - mask)))
    # A raised cosine passes through 0.5 mid-band, where the defect is 0.25.
    assert idempotency_defect > 0.2


def test_masks_agree_outside_the_taper_band(kx):
    """The taper only changes the mask inside its own band.

    That band lies wholly below the cutoff, spanning ``(cutoff - taper,
    cutoff)``, so the two edges agree over the whole retained interior and over
    everything at or above the cutoff, where both are zero.
    """
    cutoff, taper = 2 / 3, 0.1
    binary = antialias_mask(kx, cutoff=cutoff, edge="binary")
    tapered = antialias_mask(kx, cutoff=cutoff, edge="tapered", taper=taper)

    k_c = cutoff * _nyquist(kx)
    band = taper * _nyquist(kx)
    outside = (np.abs(kx) <= k_c - band) | (np.abs(kx) > k_c)

    np.testing.assert_allclose(tapered[outside], binary[outside])


# --- What the mask actually keeps ---


def test_binary_mask_passes_low_frequencies_and_blocks_high(kx):
    mask = antialias_mask(kx, cutoff=2 / 3)
    k_c = (2 / 3) * _nyquist(kx)

    assert np.all(mask[np.abs(kx) <= k_c] == 1.0)
    assert np.all(mask[np.abs(kx) > k_c] == 0.0)


@pytest.mark.parametrize("cutoff,taper", [(2 / 3, 0.2), (2 / 3, 0.05), (0.5, 0.5)])
def test_tapered_mask_transmits_nothing_above_the_cutoff(kx, cutoff, taper):
    """The roll-off lies inside the cutoff, which is what saves the 2/3 rule.

    A roll-off centred on the cutoff leaks part of the band up to
    ``cutoff + taper/2``. Under a quadratic nonlinearity two such modes sum to
    ``4/3 + taper`` of Nyquist and alias back to ``2/3 - taper`` -- inside the
    band the aperture exists to protect, where nothing downstream will remove
    them. So this is not a matter of the leak being small: the transmission
    above the cutoff has to be identically zero for the fold-back argument to
    mean anything, and a raised cosine ending exactly at the cutoff gives that.
    """
    fine_kx = np.linspace(-_nyquist(kx), _nyquist(kx), 2001)
    for grid in (kx, fine_kx):
        mask = antialias_mask(grid, cutoff=cutoff, edge="tapered", taper=taper)
        above = np.abs(grid) >= cutoff * _nyquist(grid)
        assert np.any(above), "the grid should reach past the cutoff"
        np.testing.assert_array_equal(mask[above], np.zeros_like(mask[above]))


def test_a_taper_reaching_almost_to_nyquist_is_allowed(kx):
    """Nothing constrains the taper from above now that it sits below the cutoff.

    A band that would once have run past Nyquist is simply a wide roll-off
    ending just short of it, and the mask stays within ``[0, 1]``.
    """
    mask = antialias_mask(kx, cutoff=0.98, edge="tapered", taper=0.2)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    assert np.all(mask[np.abs(kx) >= 0.98 * _nyquist(kx)] == 0.0)


def test_two_thirds_cutoff_keeps_about_two_thirds_of_the_band(kx):
    """The 2/3 rule: the top third of the band is where aliases land."""
    fraction_kept = float(np.mean(antialias_mask(kx, cutoff=2 / 3)))
    assert fraction_kept == pytest.approx(2 / 3, abs=0.05)


def test_cutoff_of_one_keeps_everything(kx):
    np.testing.assert_array_equal(antialias_mask(kx, cutoff=1.0), np.ones_like(kx))


def test_mask_is_even_in_kx(kx):
    """A real, even mask is what makes the band-limited operator self-adjoint.

    Stated as ``m(kx) == m(-kx)`` rather than as a symmetry of the sorted
    array: for even ``N`` the FFT grid carries ``-k_nyquist`` but not
    ``+k_nyquist``, so the grid itself is not symmetric about zero.
    """
    for edge, taper in (("binary", 0.0), ("tapered", 0.1)):
        mask = antialias_mask(kx, edge=edge, taper=taper)
        reflected = antialias_mask(-kx, edge=edge, taper=taper)
        np.testing.assert_allclose(mask, reflected, atol=1e-12)


def test_mask_is_real_and_within_the_unit_interval(kx):
    mask = antialias_mask(kx, edge="tapered", taper=0.2)
    assert not np.iscomplexobj(mask)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)


def test_tapered_mask_is_continuous(kx):
    """No jump at the band edges: that is what the taper buys over a hard cut."""
    fine_kx = np.linspace(-_nyquist(kx), _nyquist(kx), 2001)
    mask = antialias_mask(fine_kx, cutoff=2 / 3, edge="tapered", taper=0.2)
    assert float(np.max(np.abs(np.diff(mask)))) < 0.01


def test_narrowing_the_taper_approaches_the_binary_mask(kx):
    """The two edges are one family; `taper` interpolates between them."""
    binary = antialias_mask(kx, cutoff=2 / 3)
    deviations = [
        float(
            np.max(
                np.abs(
                    antialias_mask(kx, cutoff=2 / 3, edge="tapered", taper=t) - binary
                )
            )
        )
        for t in (0.4, 0.2, 0.1)
    ]
    assert deviations == sorted(deviations, reverse=True)


# --- Argument validation ---


@pytest.mark.parametrize("cutoff", [0.0, -0.1, 1.5])
def test_rejects_cutoff_outside_the_unit_interval(kx, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        antialias_mask(kx, cutoff=cutoff)


def test_rejects_tapered_edge_without_a_taper_width(kx):
    """Silently returning a binary mask for `edge="tapered"` would be a trap."""
    with pytest.raises(ValueError, match="taper"):
        antialias_mask(kx, edge="tapered", taper=0.0)


def test_rejects_a_taper_band_wider_than_the_cutoff(kx):
    """The whole roll-off sits below the cutoff, so it has to fit above zero.

    A taper wider than the cutoff would start at a negative wavenumber, and the
    clip would silently hide it: the mask would never reach 1 anywhere, quietly
    attenuating the low frequencies the aperture is supposed to leave alone.
    """
    with pytest.raises(ValueError, match="taper"):
        antialias_mask(kx, cutoff=0.2, edge="tapered", taper=0.5)


def test_rejects_an_unknown_edge(kx):
    """The runtime guard has to hold for callers `ty` never sees.

    `edge` is annotated as a Literal, so this call is a deliberate violation of
    the annotation -- which is the only way to reach the branch. The guard
    earns its place because notebooks and configuration files reach this
    function with strings no type checker has looked at.
    """
    with pytest.raises(ValueError, match="edge"):
        antialias_mask(kx, edge="gaussian")  # ty: ignore[invalid-argument-type]


# --- Backend agnosticism ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("edge,taper", [("binary", 0.0), ("tapered", 0.1)])
def test_mask_matches_across_backends(kx, backend_name, edge, taper):
    """The torch path may be float32, so scale the tolerance to its dtype."""
    pytest.importorskip("torch", reason="the torch backend is optional")
    backend = resolve_backend(backend_name)
    kx_backend = backend.xp.asarray(kx, dtype=backend.real, device=backend.device)

    mask = antialias_mask(kx_backend, cutoff=2 / 3, edge=edge, taper=taper)
    reference = antialias_mask(kx, cutoff=2 / 3, edge=edge, taper=taper)

    host = to_numpy(mask)
    atol = float(np.finfo(host.dtype).eps) * 16
    np.testing.assert_allclose(host, reference, atol=atol)
