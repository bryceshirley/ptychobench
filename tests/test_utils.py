"""Tests for the shared array-namespace and host-transfer helpers."""

import numpy as np
import pytest

from ptychobench.utils import (
    Array,
    Backend,
    _select_device,
    resolve_backend,
    set_row,
    to_device,
    to_numpy,
    vector_norm,
)

torch = pytest.importorskip("torch", reason="the torch backend is optional")


# --- Backend resolution ---


def test_resolve_backend_numpy_is_double_precision():
    """The NumPy path is the accuracy reference, so it must stay float64."""
    backend = resolve_backend("numpy")
    assert isinstance(backend, Backend)
    assert backend.real == np.float64
    assert backend.complex == np.complex128


def test_resolve_backend_torch_precision_follows_the_device():
    """Metal has no FP64, so the MPS path must be single precision."""
    backend = resolve_backend("torch")
    if backend.device == "mps":
        assert backend.real == torch.float32
        assert backend.complex == torch.complex64
    else:
        assert backend.real == torch.float64
        assert backend.complex == torch.complex128


def test_resolve_backend_builds_arrays_in_its_own_namespace():
    """The namespace, device and dtypes have to be mutually consistent."""
    backend = resolve_backend("torch")
    x = backend.xp.zeros(4, dtype=backend.complex, device=backend.device)
    assert x.dtype == backend.complex


def test_resolve_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("tensorflow")


# --- Host transfer ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_to_numpy_returns_a_host_array(backend_name):
    """One helper covers every backend; dispatch is on the array, not a flag."""
    backend = resolve_backend(backend_name)
    x = backend.xp.asarray([1.0, 2.0, 3.0], device=backend.device)

    host = to_numpy(x)

    assert isinstance(host, np.ndarray)
    np.testing.assert_allclose(host, [1.0, 2.0, 3.0])


def test_to_numpy_detaches_a_grad_tracking_tensor():
    """`.numpy()` on a tensor that requires grad raises; `to_numpy` must not."""
    x = torch.ones(3, requires_grad=True)
    np.testing.assert_allclose(to_numpy(x), np.ones(3))


def test_to_device_is_a_no_op_for_numpy():
    x = np.ones(3)
    assert to_device(x, None) is x


def test_to_device_moves_a_torch_tensor():
    backend = resolve_backend("torch")
    moved = to_device(torch.ones(3), backend.device)
    assert str(moved.device).startswith(str(backend.device))


# --- Linear algebra shims ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_vector_norm_matches_numpy(backend_name):
    """`vector_norm` takes only the array: the namespace comes from it."""
    backend = resolve_backend(backend_name)
    values = [3.0, 4.0, 12.0]
    x = backend.xp.asarray(values, dtype=backend.real, device=backend.device)

    got = float(vector_norm(x))

    # float32 backends carry ~7 decimal digits, so scale the tolerance to the
    # dtype rather than pinning a constant that only holds in double precision.
    rtol = float(np.finfo(to_numpy(x).dtype).eps) * 16
    assert got == pytest.approx(np.linalg.norm(values), rel=rtol)


def test_vector_norm_of_a_complex_vector_is_real_and_positive():
    x = np.asarray([1 + 1j, 1 - 1j])
    assert float(vector_norm(x)) == pytest.approx(2.0)


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_set_row_writes_in_place_and_returns_the_matrix(backend_name):
    backend = resolve_backend(backend_name)
    A = backend.xp.zeros((3, 2), dtype=backend.real, device=backend.device)
    row = backend.xp.asarray([1.0, 2.0], dtype=backend.real, device=backend.device)

    out = set_row(A, 1, row)

    np.testing.assert_allclose(to_numpy(out)[1], [1.0, 2.0])
    np.testing.assert_allclose(to_numpy(out)[0], [0.0, 0.0])


# --- The structural array type ---


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_arrays_of_every_backend_satisfy_the_array_protocol(backend_name):
    backend = resolve_backend(backend_name)
    x = backend.xp.zeros(3, dtype=backend.complex, device=backend.device)
    assert isinstance(x, Array)


# --- Device selection ---
#
# Every machine runs exactly one of these branches for real, so the other two
# are reachable only by faking the probe. That is worth doing rather than
# leaving the CUDA path untested: it shipped documented-but-absent once
# already, and nobody developing on a Mac would notice.


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_the_best_available_accelerator_is_selected(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert _select_device() == expected


def test_a_cuda_device_resolves_to_double_precision():
    """Precision follows the device, and CUDA -- unlike Metal -- has FP64.

    Resolving a device name touches no hardware, so this holds on a machine
    with no NVIDIA GPU at all.
    """
    backend = resolve_backend("torch", device="cuda")

    assert backend.device == "cuda"
    assert backend.real == torch.float64
    assert backend.complex == torch.complex128


def test_resolve_backend_honours_an_explicit_device():
    """Without this there is no way to ask for torch-on-CPU on a machine that
    has an accelerator, and so no way to tell a library difference from a
    device one."""
    backend = resolve_backend("torch", device="cpu")

    assert backend.device == "cpu"
    assert backend.real == torch.float64
    assert backend.complex == torch.complex128
