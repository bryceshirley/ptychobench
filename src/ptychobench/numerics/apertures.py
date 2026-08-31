"""
Band-limiting apertures for the transverse Fourier axis.

A split-step propagator alternates a Fourier-diagonal kinetic factor with a
real-space-diagonal potential factor. The real-space multiplication broadens
the spectrum, and on a periodic grid whatever is pushed past Nyquist does not
leave -- it wraps around and reappears at low ``kx`` as an alias. A
band-limiting aperture applied each step discards that top slice of the band
before it can fold back.

The conventional choice is the 2/3 rule: keep ``|kx| <= (2/3) k_nyquist``, so
that two spectra at the cutoff sum to ``4/3`` of Nyquist and alias to ``-2/3``,
inside the discarded region. That argument only holds if the mask transmits
*nothing* above the cutoff, which is why the ``"tapered"`` roll-off lies
entirely inside it rather than straddling it.

Notes
-----
The aperture does **not** commute with the real-space potential, so applying it
changes the operator being integrated rather than the way it is evaluated.
Nothing here applies one by default: no operator passes ``aperture=`` to
:func:`~ptychobench.numerics.splitting.lie_trotter`, because the convergence
orders the benchmark measures are properties of the unapertured operators. Pass
a mask explicitly if you want one.

Only the field is band limited, never the potential.
:mod:`ptychobench.numerics.splitting` folds the mask into the kinetic factor,
so a step applies ``P K V``; masking the transmission function as a multislice
code does diverges here. The technical notes, "Band limiting and aliasing",
carry the measurements and the abTEM comparison -- including the taper-units
trap, since abTEM measures ``taper`` as a fraction of ``1/dx``, twice Nyquist,
making its default ``0.01`` equal to ``taper=0.02`` here.
"""

from __future__ import annotations

from typing import Any, Literal

from array_api_compat import array_namespace

Edge = Literal["binary", "tapered"]


def antialias_mask(
    kx: Any,
    *,
    cutoff: float = 2 / 3,
    edge: Edge = "binary",
    taper: float = 0.0,
) -> Any:
    """
    Build a band-limiting mask over a transverse Fourier axis.

    Parameters
    ----------
    kx : array_like
        Transverse Fourier coordinates, in the FFT ordering produced by
        :attr:`ptychobench.grid.SimulationGrid.kx`. Any Array API backend.
    cutoff : float, optional
        Edge of the retained band, as a fraction of the Nyquist wavenumber.
        Must lie in ``(0, 1]``. Defaults to ``2/3``, the standard anti-alias
        rule for a quadratic nonlinearity.
    edge : {"binary", "tapered"}, optional
        How the band edge is shaped. Defaults to ``"binary"``.

        ``"binary"``
            A hard cut. The mask takes only the values 0 and 1, so it is a
            true orthogonal projection: ``P @ P == P`` exactly, and applying
            the aperture twice is the same as applying it once.
        ``"tapered"``
            A raised-cosine roll-off of width ``taper`` ending at the cutoff.
            This suppresses the ringing a hard cut produces in real space, but
            it is **not** a projection -- ``P @ P != P``, with the discrepancy
            reaching 1/4 mid-band. Composing it twice is a different operator
            from composing it once.
    taper : float, optional
        Width of the roll-off band, as a fraction of the Nyquist wavenumber.
        Required and must be positive when ``edge="tapered"``; ignored when
        ``edge="binary"``. The band lies *inside* the cutoff, spanning
        ``(cutoff - taper, cutoff)``, so ``taper`` must not exceed ``cutoff``.
        Nothing above the cutoff is transmitted by either edge.

    Returns
    -------
    array
        A real-valued mask with the shape, dtype and namespace of ``kx``, with
        values in ``[0, 1]``. Multiply a spectrum by it to band-limit.

    Raises
    ------
    ValueError
        If ``cutoff`` is outside ``(0, 1]``; if ``edge`` is not one of the two
        recognised values; if ``edge="tapered"`` and ``taper`` is not
        positive; or if the taper band would run below zero.

    Notes
    -----
    The Nyquist wavenumber is taken as ``max(|kx|)`` on the supplied grid
    rather than computed from a sample spacing, so the mask depends on nothing
    but its argument. For the even ``N`` this package uses these agree exactly
    (``max|kx| = pi/dx``, from the unpaired negative-frequency bin); for odd
    ``N`` the grid maximum falls short of ``pi/dx`` by a factor ``(N-1)/N`` and
    the retained band is correspondingly narrower.

    The mask is real and even in ``kx``. That matters downstream: a real, even
    Fourier multiplier makes the band-limited operator self-adjoint, which is
    what lets the adjoint identity be used as a correctness check.

    Examples
    --------
    >>> import numpy as np
    >>> from ptychobench.grid import SimulationGrid
    >>> mask = antialias_mask(SimulationGrid(N=64).kx)
    >>> bool(np.all(mask * mask == mask))  # a projection
    True
    """
    if not 0.0 < cutoff <= 1.0:
        raise ValueError(f"cutoff must lie in (0, 1], got {cutoff!r}")
    if edge not in ("binary", "tapered"):
        raise ValueError(f"unknown edge {edge!r}; expected 'binary' or 'tapered'")
    if edge == "tapered":
        if taper <= 0.0:
            raise ValueError(
                f'edge="tapered" needs a positive taper width, got {taper!r}; '
                'use edge="binary" for a hard cut'
            )
        if taper > cutoff:
            raise ValueError(
                f"the taper band ({cutoff - taper!r} to {cutoff!r}) runs below "
                f"zero; taper {taper!r} must not exceed cutoff {cutoff!r}"
            )

    xp = array_namespace(kx)
    k_abs = xp.abs(kx)
    k_nyquist = xp.max(k_abs)
    k_cut = cutoff * k_nyquist

    if edge == "binary":
        # astype rather than a multiply so the result is real even when the
        # comparison yields a boolean array.
        return xp.astype(k_abs <= k_cut, kx.dtype)

    # The roll-off ends *at* the cutoff rather than straddling it, so that
    # nothing above the cutoff is transmitted and the 2/3 fold-back argument
    # in the module docstring still holds. Position within it: 0 at the inner
    # edge, 1 at the cutoff, clipped flat to 1 and 0 on either side.
    band = taper * k_nyquist
    t = xp.clip((k_abs - (k_cut - band)) / band, 0.0, 1.0)
    return 0.5 * (1.0 + xp.cos(xp.pi * t))
