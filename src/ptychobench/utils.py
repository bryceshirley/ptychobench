"""
Shared structural types and backend helpers for the propagation toolbox.

Every numerical module here is written against the Array API standard rather
than against NumPy, so the same code runs on NumPy (the double-precision
accuracy reference) and on torch (including Apple Silicon MPS, which is single
precision only). This module holds the small number of shims that need to know
which backend they are talking to; nothing else in the package should.

Two conventions the rest of the package relies on:

* Get a namespace with ``array_api_compat.array_namespace(x)``, imported
  directly -- there is deliberately no wrapper for it here. Never call
  ``x.__array_namespace__()``: it is in the specification, but ``torch.Tensor``
  does not implement it, which is the reason array-api-compat exists.
* ``xp.linalg`` and ``xp.fft`` are *optional* extensions in the standard. Guard
  any use of them, as :func:`vector_norm` does, or document the requirement.

The Array API standard deliberately does not ship a runtime Protocol for
arrays (see data-apis.org/array-api/latest/design_topics/static_typing.html),
so the hand-rolled :class:`Array` below is the sanctioned approach. The rule
that keeps it honest: it may only contain members defined by *every* backend we
support. Anything torch-specific belongs behind a narrowing check, not in here.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from array_api_compat import array_namespace, is_cupy_array, is_torch_array

logger = logging.getLogger(__name__)

# torch is imported lazily so the NumPy path does not hard-depend on it.
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the installed extras
    _HAS_TORCH = False


def _select_device() -> str:
    """
    Return the best available torch device name, ``"cpu"`` as the fallback.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Notes
    -----
    CUDA is tried before MPS only because no machine has both; the order
    carries no claim about which is faster. Any given host takes exactly one of
    these branches, so the test suite fakes the two probes to cover the rest.
    """
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


#: The torch device the GPU path will build on. Consulted only by
#: :func:`resolve_backend` for ``backend="torch"``; the NumPy path is float64
#: regardless of what hardware happens to be present, so that a benchmark
#: number does not change meaning depending on the machine it ran on.
DEVICE = _select_device()


class Backend(NamedTuple):
    """
    A resolved backend: namespace plus the device and dtypes to build on.

    Attributes
    ----------
    xp : module
        The array namespace, always an ``array_api_compat`` wrapper.
    device : Any
        The device to pass to array constructors, or ``None`` for backends
        that have no such concept.
    real : Any
        The real floating dtype for this backend and device.
    complex : Any
        The complex dtype matching ``real`` in precision.
    """

    xp: Any
    device: Any
    real: Any
    complex: Any


@runtime_checkable
class Array(Protocol):
    """
    A structural type for the array operations this package actually uses.

    Every member here is part of the Array API specification, so NumPy arrays,
    torch tensors and CuPy arrays all satisfy it.

    Notes
    -----
    Use it on *parameters*, not on return types. It is deliberately a lower
    bound, so annotating a return with it promises callers almost nothing and
    breaks ordinary uses like ``np.sum(result)``. Functions returning a backend
    array should return ``Any``: you get back whatever backend you passed in.

    ``cpu()`` and ``clone()`` are deliberately absent as torch-only -- requiring
    them would mean a NumPy array fails the Protocol, which is backwards.
    ``__array_namespace__`` is absent for the reason in the module docstring.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array's shape as a tuple of dimension lengths."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        ...

    @property
    def dtype(self) -> Any:
        """Return the array's element type, in the backend's own vocabulary."""
        ...

    @property
    def device(self) -> Any:
        """Return the device the array lives on, or the backend's stand-in."""
        ...

    def __getitem__(self, key: Any, /) -> Array: ...

    # float()/complex() are called on the 0-d scalars that come out of the
    # Arnoldi recurrence and its convergence estimate.
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...

    def __abs__(self) -> Array: ...
    def __neg__(self) -> Array: ...
    def __add__(self, other: Any, /) -> Array: ...
    def __radd__(self, other: Any, /) -> Array: ...
    def __sub__(self, other: Any, /) -> Array: ...
    def __rsub__(self, other: Any, /) -> Array: ...
    def __mul__(self, other: Any, /) -> Array: ...
    def __rmul__(self, other: Any, /) -> Array: ...
    def __truediv__(self, other: Any, /) -> Array: ...
    def __rtruediv__(self, other: Any, /) -> Array: ...
    def __pow__(self, other: Any, /) -> Array: ...


def to_numpy(arr: Any) -> np.ndarray:
    """
    Copy an array from any supported backend to a host NumPy array.

    Dispatch is on the array itself, not on a namespace argument: knowing that
    the namespace is torch tells a type checker nothing about the type of
    ``arr``, and passing the two separately invites them to disagree.

    Parameters
    ----------
    arr : Any
        An array from any supported backend, or anything ``np.asarray``
        accepts.

    Returns
    -------
    numpy.ndarray
        A host array. Detached from any autograd graph, so this is a leaf.
    """
    if is_torch_array(arr):
        return arr.detach().cpu().numpy()
    if is_cupy_array(arr):
        return arr.get()
    return np.asarray(arr)


def to_device(array: Any, device: Any) -> Any:
    """
    Move an array to a device, for backends that have the concept.

    Parameters
    ----------
    array : Any
        The array to move.
    device : Any
        The target device. Ignored for backends without device placement.

    Returns
    -------
    Any
        The array on ``device``, or ``array`` unchanged for backends -- NumPy
        among them -- where placement is meaningless.
    """
    if is_torch_array(array):
        return array.to(device)
    return array


def vector_norm(x: Any) -> Any:
    """
    Return the Euclidean norm ``||x||_2`` of a vector.

    Parameters
    ----------
    x : Any
        A one-dimensional array from any supported backend. May be complex,
        in which case the result is real and non-negative.

    Returns
    -------
    Any
        A 0-d array in the namespace of ``x``.

    Notes
    -----
    ``linalg`` is an optional extension in the Array API standard, so this
    falls back to ``sqrt(sum(abs(x)**2))`` when the namespace does not provide
    it. The fallback squares the magnitudes and so overflows at the square root
    of the dtype's maximum, where ``linalg.vector_norm`` implementations
    typically rescale; that is acceptable for the unit-normalised wavefields
    this package propagates.
    """
    xp = array_namespace(x)
    if hasattr(xp, "linalg"):
        return xp.linalg.vector_norm(x)
    return xp.sqrt(xp.sum(xp.abs(x) ** 2))


def set_row(A: Any, i: int, row: Any) -> Any:
    """
    Write ``row`` into row ``i`` of ``A``, in place where possible.

    Parameters
    ----------
    A : Any
        A two-dimensional array.
    i : int
        The row index to write.
    row : Any
        The values to write, broadcastable to ``A[i, :]``.

    Returns
    -------
    Any
        The updated matrix. Callers must use the return value rather than
        relying on mutation: immutable backends such as JAX return a new
        array, and the in-place branch returns ``A`` itself so both read the
        same at the call site.
    """
    if hasattr(A, "at"):
        return A.at[i, :].set(row)
    A[i, :] = row
    return A


@lru_cache(maxsize=1)
def _announce_mps_precision() -> None:
    """
    Say once per process that the MPS path is single precision.

    Cached because :func:`resolve_backend` is cheap and gets called freely --
    once per grid, and a script that sweeps configurations calls it many times.
    Repeating the same notice on every call would bury whatever the caller
    actually wanted to see.
    """
    logger.info("Using Apple Silicon GPU (MPS); Metal has no FP64, so float32.")


def resolve_backend(backend: str = "numpy", device: Any = None) -> Backend:
    """
    Map a backend name onto its namespace, device and dtypes.

    This is the single place where a string turns into a namespace. Everything
    downstream takes an ``xp`` and never asks which library it is.

    Parameters
    ----------
    backend : str, optional
        One of ``"numpy"``, ``"torch"`` or ``"cupy"``. Defaults to
        ``"numpy"``, which is the double-precision accuracy reference.
    device : Any, optional
        The device to build on, e.g. ``"cpu"``, ``"mps"`` or ``"cuda"``.
        ``None``, the default, means the backend's own choice -- for torch that
        is :data:`DEVICE`, the best accelerator present. Pass ``"cpu"``
        explicitly to compare a GPU run against the same library on the host,
        which separates a backend difference from a device difference.

        Ignored by backends where placement is meaningless.

    Returns
    -------
    Backend
        The namespace, device and dtypes to build arrays with. **The dtypes
        follow the device**, not just the library: see below.

    Raises
    ------
    ImportError
        If the requested backend is known but not installed.
    ValueError
        If ``backend`` is not one of the recognised names.
    """
    if backend == "torch":
        if not _HAS_TORCH:
            raise ImportError('backend="torch" requires torch to be installed')
        import array_api_compat.torch as xp

        resolved = DEVICE if device is None else device
        if str(resolved).startswith("mps"):
            # Metal has no FP64, so the GPU path is single precision. Tests
            # that compare against it must scale tolerances to the dtype. This
            # is a property of the *device*, which is why torch-on-CPU below
            # stays double even though it is the same library.
            _announce_mps_precision()
            return Backend(xp, resolved, torch.float32, torch.complex64)
        return Backend(xp, resolved, torch.float64, torch.complex128)

    # The dtypes come from `np` rather than from `xp` because array-api-compat
    # re-exports its wrapped namespaces with a star import, so the attributes
    # exist at runtime but are invisible to static analysis. CuPy uses NumPy's
    # dtype objects too, so one source serves both.
    if backend == "numpy":
        import array_api_compat.numpy as xp

        return Backend(xp, None, np.float64, np.complex128)

    if backend == "cupy":
        import array_api_compat.cupy as xp

        return Backend(xp, device, np.float64, np.complex128)

    raise ValueError(
        f"unknown backend {backend!r}; expected 'numpy', 'torch' or 'cupy'"
    )
