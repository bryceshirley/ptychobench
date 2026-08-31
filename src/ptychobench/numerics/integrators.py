"""
Matrix-free evaluation of ``f(A) v`` by Krylov subspace projection.

The dense path in :mod:`ptychobench.operators` forms the ``N x N`` propagation
operator and hands it to :func:`scipy.linalg.expm` or :func:`scipy.linalg.sqrtm`.
That is the accuracy reference and it stays, but it costs ``O(N^2)`` storage and
``O(N^3)`` work per step, and it needs the operator as a matrix -- which the
Fourier-diagonal factors are not, naturally.

Arnoldi projection removes both constraints. Given only the *action* of ``A``
on a vector, it builds an orthonormal basis ``V_m`` of the Krylov subspace
``span{v, Av, ..., A^(m-1) v}`` and the small upper Hessenberg matrix
``H_m = V_m^H A V_m``, then approximates

    f(A) v  ~  ||v|| * V_m * f(H_m) * e_1

with ``m`` typically a few tens regardless of ``N``. ``f`` is still evaluated
densely, but on ``H_m`` rather than on ``A``.

Conventions this module fixes, each of which a caller can otherwise get wrong:

* **``f`` is a host function.** ``H_m`` is built and kept as a NumPy array on
  the host, because ``scipy.linalg.expm``/``sqrtm`` are host-only. ``f`` takes
  and returns a host matrix; the basis stays on whatever device the input
  vector came from.
* **Shape survives the round trip.** Krylov projection cares about the vector
  space, not how it is indexed. ``v`` may have any shape; the recurrence runs
  on the flattened vector and the result is reshaped back. ``matvec`` is always
  called with ``v``'s own shape, never the flattened one, so an operator built
  around a 2D transform keeps working unchanged.
* **``tol`` is relative to ``||v||``** -- see :func:`krylov_funm`.

References
----------
.. [1] Y. Saad, "Analysis of some Krylov subspace approximations to the matrix
       exponential operator", SIAM J. Numer. Anal. 29(1), 209-228 (1992).
.. [2] E. Gallopoulos and Y. Saad, "Efficient solution of parabolic equations
       by Krylov approximation methods", SIAM J. Sci. Stat. Comput. 13(5),
       1236-1264 (1992).
.. [3] M. Hochbruck and C. Lubich, "On Krylov subspace approximations to the
       matrix exponential operator", SIAM J. Numer. Anal. 34(5), 1911-1925
       (1997).
.. [4] N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM
       (2008). Chapter 6 on the square root and its branch cut.
.. [5] L. Giraud, J. Langou and M. Rozloznik, "The loss of orthogonality in the
       Gram-Schmidt orthogonalization process", Comput. Math. Appl. 50,
       1069-1075 (2005). The result behind the two-pass scheme used here.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple, Protocol

import numpy as np
from array_api_compat import array_namespace

from ..utils import Array, set_row, to_numpy, vector_norm

#: Host matrix in, host matrix out. ``scipy.linalg.expm`` and
#: ``scipy.linalg.sqrtm`` both satisfy this.
MatrixFunction = Callable[[np.ndarray], np.ndarray]

#: The action of the operator on a field, in the caller's own array shape.
MatVec = Callable[[Any], Any]


class StopTest(Protocol):
    """The early-exit predicate :func:`arnoldi` consults after each step."""

    def __call__(self, hessenberg: np.ndarray, h_next: float, step: int, /) -> bool:
        """
        Decide whether the recurrence has converged.

        Parameters
        ----------
        hessenberg : numpy.ndarray
            The active ``(step, step)`` block of ``H``, on the host.
        h_next : float
            The subdiagonal entry ``h_{step+1,step}``, i.e. the norm of the
            residual that step produced.
        step : int
            The number of basis vectors built so far.

        Returns
        -------
        bool
            True to stop the recurrence and keep the current subspace.
        """
        ...


class ArnoldiResult(NamedTuple):
    """
    The output of an Arnoldi run.

    Attributes
    ----------
    basis : Any
        The orthonormal basis, ``(size + 1, n)`` with one basis vector **per
        row** and ``n`` the flattened length of the input. Rows ``0 .. size-1``
        span the Krylov subspace; row ``size`` holds the next, un-included
        vector ``v_{m+1}``, which is all zeros on breakdown. On the same device
        and in the same namespace as the input vector.
    hessenberg : numpy.ndarray
        The ``(size + 1, size)`` upper Hessenberg matrix, complex128, on the
        **host** -- ``f`` is a host function, so keeping ``H`` there avoids a
        device round trip per iteration.
    size : int
        The dimension ``m`` of the subspace actually built. Less than
        ``max_iter`` if the recurrence broke down or ``stop`` fired.
    beta : float
        ``||v||``, the norm of the input vector.
    breakdown : bool
        True if the recurrence terminated because the subspace became
        ``A``-invariant. This is *success*: the projection is then exact.

    Notes
    -----
    The defining identity, verified in the test suite, is

        ``A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T``

    with ``V_m = basis[:size].T`` and ``H_m = hessenberg[:size, :size]``. On
    breakdown the trailing term vanishes identically and ``A V_m = V_m H_m``.
    """

    basis: Any
    hessenberg: np.ndarray
    size: int
    beta: float
    breakdown: bool


def arnoldi(
    matvec: MatVec,
    v: Array,
    max_iter: int,
    *,
    stop: StopTest | None = None,
) -> ArnoldiResult:
    """
    Build an orthonormal Krylov basis for ``A`` and ``v`` by Arnoldi.

    Parameters
    ----------
    matvec : callable
        The action ``x -> A x``. Called with arrays of ``v``'s own shape, so an
        operator written against a 2D field needs no adaptation.
    v : Array
        The starting vector, of any shape. Only its flattened contents matter
        to the recurrence.
    max_iter : int
        The maximum subspace dimension. Silently capped at the length of the
        flattened vector, beyond which the Krylov space cannot grow.
    stop : StopTest, optional
        Consulted after each completed step; see :class:`StopTest`. Exceptions
        raised by it are **not** caught. Defaults to running to ``max_iter``.

    Returns
    -------
    ArnoldiResult
        The basis, the Hessenberg matrix, and how the recurrence terminated.

    Notes
    -----
    **Orthogonalisation.** Classical Gram-Schmidt applied twice (CGS2). One
    pass leaves the basis orthogonal only to about ``sqrt(eps)``; a second pass
    recovers ``eps`` and is known to be enough [5]_. Both passes are written as
    two matrix-vector products against the whole basis block rather than as a
    Python loop over basis vectors, which is what makes the cost per step a
    pair of BLAS-2 calls instead of ``j`` BLAS-1 calls.

    **Breakdown.** The step normalises by ``h_{m+1,m}``, which is zero exactly
    when the subspace is ``A``-invariant. The guard threshold is
    ``n * eps * ||A v_j||``: it has to scale with the local operator norm
    because ``A`` may be scaled arbitrarily, and with ``eps`` of the working
    dtype because the MPS path is single precision. Below that threshold the
    residual is indistinguishable from the round-off in forming it, so no
    direction can be recovered from it. The result is flagged
    ``breakdown=True`` and is *exact*, not degraded.
    """
    xp = array_namespace(v)
    shape = tuple(v.shape)
    v_flat = xp.reshape(v, (-1,))
    n = int(v_flat.shape[0])
    beta = float(vector_norm(v_flat))

    # The Krylov space of a length-n vector cannot exceed dimension n.
    m_max = max(0, min(int(max_iter), n))

    if beta == 0.0 or m_max == 0:
        # span{0, A0, ...} is the zero subspace. Return it rather than dividing
        # by beta; krylov_funm short-circuits this case to an exact zero.
        return ArnoldiResult(
            basis=xp.zeros((1, n), dtype=v_flat.dtype, device=v_flat.device),
            hessenberg=np.zeros((1, 0), dtype=np.complex128),
            size=0,
            beta=beta,
            breakdown=True,
        )

    v0 = v_flat / beta

    # The first product is taken before allocating so the basis dtype can be
    # promoted to hold it: a real starting vector under a complex operator
    # still needs a complex basis for the recurrence to close.
    w = xp.reshape(matvec(xp.reshape(v0, shape)), (-1,))
    dtype = xp.result_type(v_flat.dtype, w.dtype)
    eps = float(xp.finfo(dtype).eps)

    basis = xp.zeros((m_max + 1, n), dtype=dtype, device=v_flat.device)
    basis = set_row(basis, 0, v0)

    # complex128 regardless of the backend: H is m x m with m in the tens, so
    # host double precision is free here and strictly better for scipy's f.
    hessenberg = np.zeros((m_max + 1, m_max), dtype=np.complex128)

    size = m_max
    breakdown = False

    for j in range(m_max):
        if j > 0:
            w = xp.reshape(matvec(xp.reshape(basis[j, :], shape)), (-1,))

        # ||A v_j||, the local scale the breakdown threshold is measured
        # against. Captured before orthogonalisation, which only shrinks it.
        scale = float(vector_norm(w))

        block = basis[: j + 1, :]
        h1 = xp.conj(block) @ w
        w = w - (h1 @ block)
        h2 = xp.conj(block) @ w
        w = w - (h2 @ block)
        hessenberg[: j + 1, j] = to_numpy(h1 + h2)

        h_next = float(vector_norm(w))
        if h_next <= n * eps * scale:
            # Happy breakdown: span{v, ..., A^j v} is A-invariant, so f(A)v
            # lies in it exactly. Leave h_{j+1,j} and v_{j+1} at zero, which
            # keeps the Arnoldi identity true rather than merely unchecked.
            size = j + 1
            breakdown = True
            break

        hessenberg[j + 1, j] = h_next
        basis = set_row(basis, j + 1, w / h_next)

        if stop is not None and stop(hessenberg[: j + 1, : j + 1], h_next, j + 1):
            size = j + 1
            break

    return ArnoldiResult(
        basis=basis,
        hessenberg=hessenberg,
        size=size,
        beta=beta,
        breakdown=breakdown,
    )


def _make_stop_test(f: MatrixFunction, tol: float) -> StopTest:
    """
    Build the a-posteriori stopping predicate for :func:`krylov_funm`.

    Parameters
    ----------
    f : callable
        The matrix function, applied to the projected Hessenberg matrix.
    tol : float
        Convergence tolerance, already divided through by ``||v||``.

    Returns
    -------
    StopTest
        A predicate matching the :class:`StopTest` protocol.
    """

    def stop(hessenberg: np.ndarray, h_next: float, step: int, /) -> bool:
        """
        Report whether the error estimate has dropped below ``tol``.

        Parameters
        ----------
        hessenberg : numpy.ndarray
            The active ``(step, step)`` block of ``H``.
        h_next : float
            The subdiagonal entry ``h_{step+1,step}``.
        step : int
            The number of basis vectors built so far.

        Returns
        -------
        bool
            True once the a-posteriori estimate is within tolerance.

        Raises
        ------
        FloatingPointError
            If ``f(H)`` produces a non-finite entry, which would otherwise
            compare False against any tolerance forever.
        """
        # Not wrapped in try/except on purpose: a failure inside f(H) is a real
        # failure, and swallowing it here would silently disable the early exit
        # and burn every remaining iteration for nothing.
        fH = f(hessenberg)
        estimate = h_next * abs(complex(fH[step - 1, 0]))
        if not np.isfinite(estimate):
            # A nan compares False against any tolerance, which is
            # indistinguishable from "not converged yet" -- forever.
            raise FloatingPointError(
                f"the Krylov convergence estimate is not finite ({estimate}) at "
                f"step {step}; f(H) produced a non-finite entry"
            )
        return bool(estimate <= tol)

    return stop


def krylov_funm(
    matvec: MatVec,
    v: Array,
    f: MatrixFunction,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> Any:
    """
    Approximate ``f(A) v`` without forming ``A``.

    Parameters
    ----------
    matvec : callable
        The action ``x -> A x``. Called with arrays of ``v``'s own shape.
    v : Array
        The vector to apply ``f(A)`` to, of any shape.
    f : callable
        The matrix function, applied to the small projected Hessenberg matrix.
        Takes and returns a **host** NumPy matrix -- ``scipy.linalg.expm`` and
        ``scipy.linalg.sqrtm`` are the intended arguments and are host-only.
        Exceptions from ``f`` propagate.
    max_iter : int, optional
        The maximum subspace dimension, default 30. Capped at the length of the
        flattened ``v``.
    tol : float, optional
        Convergence tolerance, default ``1e-8``, **relative to ``||v||``**. Pass
        ``tol <= 0`` to disable the early exit and always take ``max_iter``
        steps; ``f`` is then evaluated once, at the end.

    Returns
    -------
    Any
        ``f(A) v``, in the namespace, dtype and device of ``v`` and with ``v``'s
        shape.

    Raises
    ------
    FloatingPointError
        If the convergence estimate is not finite, which means ``f(H)``
        produced a nan or an inf.

    Notes
    -----
    **The stopping criterion.** The a-posteriori estimate of the absolute error
    is ``||v|| * h_{m+1,m} * |f(H_m)_{m,1}|`` [1]_; dividing through by
    ``||v||`` gives the quantity compared against ``tol``, which is why ``tol``
    is relative and why scaling ``v`` does not change the iteration count.

    The estimate is only *first order*: it is the leading term of the residual
    expansion and is reliable for normal ``A``, but can under-predict the true
    error substantially when ``A`` is far from normal, which the band-limited
    operators here are. Treat it as a stopping heuristic, not as an error bound.
    Where the answer has to be trusted, compare against the dense path in
    :mod:`ptychobench.operators`, which is the accuracy reference.

    Evaluating the estimate costs one dense ``f`` on an ``m x m`` matrix per
    step. That is cheap next to ``f`` on the full operator but not free, so
    ``tol <= 0`` skips it entirely.
    """
    xp = array_namespace(v)

    stop = _make_stop_test(f, tol) if tol > 0 else None
    result = arnoldi(matvec, v, max_iter, stop=stop)
    if result.size == 0:
        return xp.zeros_like(v)

    m = result.size
    fH = f(result.hessenberg[:m, :m])

    # beta * V_m f(H_m) e_1, with the basis stored one vector per row.
    y = np.asarray(fH)[:m, 0] * result.beta
    basis = result.basis[:m, :]
    y_backend = xp.asarray(y, dtype=basis.dtype, device=basis.device)

    return xp.reshape(y_backend @ basis, tuple(v.shape))


def warn_if_near_branch_cut(
    matrix: Any,
    *,
    context: str,
    rtol: float | None = None,
) -> bool:
    """
    Warn if a matrix has eigenvalues near the branch cut of ``sqrt``.

    The principal square root has its cut on the negative real axis, where it
    is discontinuous: an eigenvalue at ``-a + i*delta`` and one at
    ``-a - i*delta`` have principal roots of opposite sign no matter how small
    ``delta`` is. Clipping evanescent modes -- setting ``kz = 0`` for
    ``|kx| > k0``, as :meth:`ptychobench.grid.SimulationGrid.get_kinetic_operator`
    does -- pushes eigenvalues of ``1 + mu + E`` onto exactly that axis, so
    ``sqrtm`` of the projected matrix can pick a side essentially at random.

    Parameters
    ----------
    matrix : Any
        A square matrix, from any backend. Copied to the host to take
        eigenvalues.
    context : str
        Named in the warning so the caller can be identified from the message.
    rtol : float, optional
        Relative width of the band around the negative real axis that counts as
        "near". Defaults to ``sqrt(eps)`` of the matrix dtype, scaled by the
        spectral radius. The default is not a statement about the mathematics
        but about what is *knowable*: the projection itself carries error of
        order ``eps * ||A||``, so an imaginary part at that level does not
        determine which side of the cut the eigenvalue is on.

    Returns
    -------
    bool
        True if at least one eigenvalue was flagged, in which case a
        :exc:`RuntimeWarning` was issued. False otherwise, with no warning.

    Notes
    -----
    Only eigenvalues with a *strictly negative* real part are flagged. An
    eigenvalue at exactly zero is the branch point rather than the cut, and the
    principal root is continuous there, so a fully clipped mode is not a
    hazard. An eigenvalue with a negative real part but a substantial imaginary
    part is well clear of the cut and is also not flagged.

    This warns rather than raising: a clipped spectrum is a normal consequence
    of band-limiting, and the caller may legitimately want the result anyway.
    """
    host = to_numpy(matrix)
    if host.size == 0:
        return False

    eigenvalues = np.linalg.eigvals(host)
    if rtol is None:
        # result_type guards against an integer input, which has no finfo.
        rtol = float(np.sqrt(np.finfo(np.result_type(host.dtype, np.float32)).eps))

    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0.0:
        return False

    near_cut = (np.real(eigenvalues) < 0.0) & (
        np.abs(np.imag(eigenvalues)) <= rtol * scale
    )
    if not bool(np.any(near_cut)):
        return False

    flagged = eigenvalues[near_cut]
    warnings.warn(
        f"{context}: {flagged.size} eigenvalue(s) lie near the branch cut of "
        f"sqrt on the negative real axis (e.g. {flagged[0]:.6g}); the principal "
        "square root is discontinuous there, so its sign is not determined by "
        "the data. This usually means evanescent modes have been clipped.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True
