"""Float64 Cholesky helpers for dgpsi.

L is a valid square lower Cholesky factor. Only its leading k rows/columns
are used. RHS buffers must be writable and must not overlap L. Factors may
be C, F or strided; each call owns its LAPACK argument/work arrays.
"""
import ctypes
import numpy as np
from numba import njit
from numba.extending import get_cython_function_address

_byte_p = ctypes.POINTER(ctypes.c_uint8)
_int_p = ctypes.POINTER(ctypes.c_int32)
_double_p = ctypes.POINTER(ctypes.c_double)


def _resolve(name, arguments):
    try:
        address = get_cython_function_address("scipy.linalg.cython_lapack", name)
        return ctypes.CFUNCTYPE(None, *arguments)(address), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


dpotrs_ct, DPOTRS_ERROR = _resolve(
    "dpotrs", [_byte_p, _int_p, _int_p, _double_p, _int_p,
               _double_p, _int_p, _int_p])



def _probe():
    """Exercise C/upper and F/lower storage without compiling at import."""
    if dpotrs_ct is None:
        return False, DPOTRS_ERROR
    try:
        a = np.array([[4., 1.], [1., 3.]])
        for order, triangle in (("C", ord("U")), ("F", ord("L"))):
            factor = np.array(np.linalg.cholesky(a), order=order)
            rhs = np.array([[1., 2.], [3., 4.]], order="F")
            expected = np.linalg.solve(a, rhs)
            uplo = np.array([triangle], dtype=np.uint8)
            args = np.array([2, 2, 2, 2, 0], dtype=np.int32)
            pointers = [args[i:].ctypes.data_as(_int_p) for i in range(5)]
            dpotrs_ct(uplo.ctypes.data_as(_byte_p), pointers[0], pointers[1],
                      factor.ctypes.data_as(_double_p), pointers[2],
                      rhs.ctypes.data_as(_double_p), pointers[3], pointers[4])
            if args[4] != 0 or not np.allclose(rhs, expected, rtol=1e-12, atol=1e-12):
                return False, "LAPACK self-test failed"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


LAPACK_CHO_SOLVE_ENABLED, DPOTRS_ERROR = _probe()


@njit(cache=False)
def _dpotrs_inplace_from_L(L, B, k, lower=True):
    """Private raw bridge: contiguous square L, writable F-order B, validated sizes.

For raw C-order lower L use lower=False: LAPACK sees its upper transpose.
Only B[:k, :] is overwritten. No pointer-bearing function is disk-cached.
"""
    uplo = np.empty(1, np.uint8)
    uplo[0] = 76 if lower else 85
    args = np.empty(5, np.int32)
    args[0], args[1], args[2], args[3], args[4] = k, B.shape[1], L.shape[0], B.shape[0], 0
    dpotrs_ct(uplo.ctypes, args[0:].ctypes, args[1:].ctypes, L.ctypes,
              args[2:].ctypes, B.ctypes, args[3:].ctypes, args[4:].ctypes)
    return args[4]


@njit(cache=True, fastmath=True)
def forward_solve_inplace(L, b, k):
    """Trusted internal loop: overwrite b[:k] with the solution of L x=b."""
    for i in range(k):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * b[j]
        b[i] = s / L[i, i]


@njit(cache=True, fastmath=True)
def backward_solve_inplace(L, b, k):
    """Trusted internal loop: overwrite b[:k] with the solution of L.T x=b."""
    for i in range(k - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, k):
            s -= L[j, i] * b[j]
        b[i] = s / L[i, i]


@njit(cache=True)
def chol_solve_inplace(L, b, k):
    forward_solve_inplace(L, b, k)
    backward_solve_inplace(L, b, k)


@njit(cache=True, inline="always")
def _check_factor(L, k):
    if not L.dtype == np.dtype(np.float64):
        raise ValueError("Expected a float64 factor")
    if L.shape[0] != L.shape[1] or k < 0 or k > L.shape[0]:
        raise ValueError("Expected a square factor and 0 <= k <= its size")
    if L.shape[0] > 2147483647:
        raise ValueError("Factor exceeds LAPACK int32 dimensions")


@njit(cache=True, inline="always")
def _overlaps(L, b):
    """Conservative byte spans also cover negative/strided views."""
    if L.size == 0 or b.size == 0:
        return False
    l0 = np.int64(L.ctypes.data)
    b0 = np.int64(b.ctypes.data)
    s0 = (L.shape[0] - 1) * L.strides[0]
    s1 = (L.shape[1] - 1) * L.strides[1]
    sb = (b.shape[0] - 1) * b.strides[0]
    lo = l0 + min(0, s0) + min(0, s1)
    hi = l0 + max(0, s0) + max(0, s1) + L.itemsize
    return lo < b0 + max(0, sb) + b.itemsize and b0 + min(0, sb) < hi


@njit(cache=False)
def chol_solve_vec_inplace(L, alpha, k):
    """Solve (L L.T) x=alpha[:k], preserving the tail. Return 0 LAPACK, 1 loop.

Float64 arrays are required. Unsupported types are rejected during Numba
typing or validation; size/overlap errors raise ValueError before writing.
"""
    _check_factor(L, k)
    if not alpha.dtype == np.dtype(np.float64):
        raise ValueError("Expected a float64 RHS")
    if alpha.size < k:
        raise ValueError("RHS is shorter than k")
    if _overlaps(L, alpha):
        raise ValueError("Factor and RHS must not overlap")
    if k == 0:
        return 1
    if LAPACK_CHO_SOLVE_ENABLED and alpha.flags.c_contiguous:
        if L.flags.c_contiguous or L.flags.f_contiguous:
            
            
            B = np.ascontiguousarray(alpha[:k]).reshape((k, 1))
            info = _dpotrs_inplace_from_L(L, B, k, lower=not L.flags.c_contiguous)
            if info != 0:
                raise ValueError("DPOTRS rejected validated arguments")
            return 0
    chol_solve_inplace(L, alpha, k)
    return 1


@njit(cache=False)
def chol_inv_eye(A, L, k):
    """Return an F-order inverse from L[:k,:k], without modifying A or L.

A is retained for signature compatibility; callers must supply its lower
Cholesky factor L. The fallback also reuses L, avoiding a new factorization.
"""
    _check_factor(L, k)
    if A.shape[0] != A.shape[1] or A.shape[0] < k:
        raise ValueError("Expected a square A at least k by k")
    inv = np.eye(k, dtype=np.float64).T
    if k == 0:
        return inv
    if LAPACK_CHO_SOLVE_ENABLED and (L.flags.c_contiguous or L.flags.f_contiguous):
        info = _dpotrs_inplace_from_L(L, inv, k, lower=not L.flags.c_contiguous)
        if info != 0:
            raise ValueError("DPOTRS rejected validated arguments")
        return inv
    for i in range(k):
        chol_solve_inplace(L, inv[:, i], k)
    return inv
