"""
The unitary transverse DFT, and application of Fourier-diagonal operators.

Every Fourier-diagonal operator in this package is built as ``F_inv @ diag(g) @
F`` with ``F`` the *unitary* DFT -- ``scipy.linalg.dft(N) / sqrt(N)``, as
:class:`ptychobench.grid.SimulationGrid` constructs it. The matrix-free path
has to use the same convention or every comparison against the dense reference
is measuring a normalisation mismatch rather than a discretisation error.

That is the whole reason this module exists rather than callers reaching for
``xp.fft.fft`` directly: the normalisation is a property of the physics
formulation, not of the backend, and it should be stated in exactly one place.

Unitarity also buys the norm-conservation test. With ``F`` unitary and a
unit-modulus Fourier multiplier, the propagated field keeps its 2-norm to
round-off, so a drift in the norm is evidence of a bug rather than of the
transform's own scaling.

The transforms are agnostic to transverse rank: they act on every axis of the
field by default, so the same code serves a 1D and a 2D transverse grid. This
package has no batch axis anywhere, which is what makes "every axis" the safe
default; a caller that does batch passes ``axes`` explicitly.

Notes
-----
``fft`` is an *optional* extension in the Array API standard: a conforming
namespace may omit it. Everything here routes through :func:`fft_namespace`,
which raises :exc:`NotImplementedError` naming the backend, so a missing
extension never surfaces as a bare ``AttributeError`` from deep inside a
propagation loop. :mod:`ptychobench.numerics.metrics` uses that guard too, for its
far-field transform, which has its own (non-unitary) normalisation and so does
not go through :func:`fft`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from array_api_compat import array_namespace

# The Array API's orthonormal scaling: 1/sqrt(N) on both the forward and the
# inverse transform. This is the one place the convention is written down.
_NORM = "ortho"


def fft_namespace(x: Any) -> Any:
    """
    Return the ``fft`` extension of ``x``'s namespace.

    Parameters
    ----------
    x : array
        Any array; only its namespace is used.

    Returns
    -------
    module
        The ``fft`` submodule of the array's namespace.

    Raises
    ------
    NotImplementedError
        If the namespace does not provide the optional ``fft`` extension.
    """
    xp = array_namespace(x)
    fft_ext = getattr(xp, "fft", None)
    if fft_ext is None:
        raise NotImplementedError(
            f"the array namespace {getattr(xp, '__name__', xp)!r} does not provide "
            "the optional 'fft' extension, so the matrix-free path is unavailable; "
            "use the dense operators instead"
        )
    return fft_ext


def fft(x: Any, axes: Sequence[int] | None = None) -> Any:
    """
    Forward unitary DFT over the transverse axes.

    Parameters
    ----------
    x : array
        A complex field sampled on the transverse grid, in real space. Every
        axis is transverse by default -- see Notes.
    axes : sequence of int, optional
        The axes to transform. Defaults to all of them.

    Returns
    -------
    array
        The spectrum in FFT bin ordering, matching
        :attr:`ptychobench.grid.SimulationGrid.kx` on each transformed axis.
        For a rank-1 field, identical to ``grid.F @ x`` up to round-off.

    Raises
    ------
    NotImplementedError
        If the array's namespace lacks the optional ``fft`` extension.

    Notes
    -----
    The default transforms *every* axis because nothing in this package carries
    a batch axis: a field's rank is its transverse rank. A caller that does
    batch must pass ``axes`` explicitly, because the wrong choice here is silent
    -- a last-axis-only transform of a 2D field is still unitary and still
    invertible, so neither Parseval nor a roundtrip would catch it.
    """
    return fft_namespace(x).fftn(x, axes=axes, norm=_NORM)


def ifft(x: Any, axes: Sequence[int] | None = None) -> Any:
    """
    Inverse unitary DFT over the transverse axes.

    Parameters
    ----------
    x : array
        A spectrum in FFT bin ordering.
    axes : sequence of int, optional
        The axes to transform. Defaults to all of them, matching :func:`fft`.

    Returns
    -------
    array
        The real-space field. For a rank-1 field, identical to
        ``grid.F_inv @ x`` up to round-off.

    Raises
    ------
    NotImplementedError
        If the array's namespace lacks the optional ``fft`` extension.
    """
    return fft_namespace(x).ifftn(x, axes=axes, norm=_NORM)


def fourier_multiply(x: Any, multiplier: Any, axes: Sequence[int] | None = None) -> Any:
    """
    Apply a Fourier-diagonal operator without forming it.

    Computes ``F_inv @ diag(multiplier) @ F @ x`` in ``O(N log N)`` rather than
    the ``O(N^2)`` of the dense product, and without the ``O(N^2)`` storage.

    Parameters
    ----------
    x : array
        A complex field in real space, of any transverse rank.
    multiplier : array
        The operator's Fourier-space diagonal, sampled on the same ``kx`` grid
        and in the same bin ordering as ``fft(x)``. May be real or complex, and
        must broadcast against ``x``.
    axes : sequence of int, optional
        The transverse axes, passed through to :func:`fft`. Defaults to all of
        them.

    Returns
    -------
    array
        The field with the operator applied, in real space.

    Raises
    ------
    NotImplementedError
        If the array's namespace lacks the optional ``fft`` extension.

    Notes
    -----
    The resulting operator is self-adjoint when ``multiplier`` is real and even
    in ``kx``, and unitary when ``multiplier`` has unit modulus. Both hold for
    the operators this package composes -- the band-limiting mask for the
    first, a free-space propagation phase for the second -- and both are
    verified in the test suite rather than assumed.
    """
    return ifft(multiplier * fft(x, axes), axes)
