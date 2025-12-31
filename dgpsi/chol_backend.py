import numpy as np
import ctypes
from numba import njit
from numba.extending import get_cython_function_address

# ------------------------------------------------------------
# 1) Resolve SciPy cython_lapack dpotrs for Numba (needs int* UPLO trick)
# ------------------------------------------------------------
_ptr_dble = ctypes.POINTER(ctypes.c_double)
_ptr_int  = ctypes.POINTER(ctypes.c_int)

try:
    dpotrs_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dpotrs")
    dpotrs_py = ctypes.CFUNCTYPE(                           # FIX
        None, ctypes.c_char_p, _ptr_int, _ptr_int, _ptr_dble, _ptr_int, _ptr_dble, _ptr_int, _ptr_int
    )(dpotrs_addr) 
    dpotrs_ct = ctypes.CFUNCTYPE(
        None, _ptr_int, _ptr_int, _ptr_int, _ptr_dble, _ptr_int, _ptr_dble, _ptr_int, _ptr_int
    )(dpotrs_addr)
    _HAVE_DPOTRS = True
except Exception:
    dpotrs_py = None
    dpotrs_ct = None
    _HAVE_DPOTRS = False


def _lapack_self_test_python() -> bool:
    """
    Import-time sanity check (no Numba compile):
      - force F-order float64 arrays
      - call dpotrs via correct ctypes signature (char* UPLO)
      - compare to np.linalg.solve
    """
    if not _HAVE_DPOTRS:
        return False

    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]], dtype=np.float64, order="F")
    L = np.array(np.linalg.cholesky(A), dtype=np.float64, order="F", copy=False)

    B = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float64, order="F")
    X_ref = np.linalg.solve(A, B)

    B2 = B.copy(order="F")

    uplo = b"L"
    n    = ctypes.c_int(2)
    nrhs = ctypes.c_int(2)
    lda  = ctypes.c_int(L.shape[0])
    ldb  = ctypes.c_int(B2.shape[0])
    info = ctypes.c_int(0)

    dpotrs_py(
        uplo,
        ctypes.byref(n),
        ctypes.byref(nrhs),
        L.ctypes.data_as(_ptr_dble),
        ctypes.byref(lda),
        B2.ctypes.data_as(_ptr_dble),
        ctypes.byref(ldb),
        ctypes.byref(info),
    )

    if info.value != 0:
        return False
    return np.allclose(B2, X_ref, rtol=1e-12, atol=1e-12)


# Global on/off flag (Python bool, frozen at import-time)
LAPACK_CHO_SOLVE_ENABLED: bool = bool(_lapack_self_test_python())

@njit(cache=False, fastmath=True)
def _dpotrs_inplace_from_L(L, B, k, lower=True):
    """
    In-place solve using LAPACK dpotrs from precomputed factor stored in L.

    Solves: A X = B for N=k, where A = L L^T (lower=True) using top-left k×k.
    Overwrites the first k rows of B (B shape is (ldb, nrhs), ldb >= k).

    Returns INFO (0 success).
    """
    UPLO = np.array(ord('L') if lower else ord('U'), dtype=np.int32)
    INFO = np.array(0, dtype=np.int32)

    N   = np.array(k, dtype=np.int32)
    LDA = np.array(L.shape[0], dtype=np.int32)

    NRHS = np.array(B.shape[1], dtype=np.int32)
    LDB  = np.array(B.shape[0], dtype=np.int32)

    dpotrs_ct(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        B.ctypes,
        LDB.ctypes,
        INFO.ctypes
    )
    return INFO[()]

@njit(cache=True, fastmath=True)
def forward_solve_inplace(L, b, k):
    """Overwrite b[:k] with x solving L x = b (L lower-triangular)."""
    for i in range(k):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * b[j]
        b[i] = s / L[i, i]

@njit(cache=True, fastmath=True)
def backward_solve_inplace(L, b, k):
    """Overwrite b[:k] with x solving (L.T) x = b, where L is lower-triangular."""
    for i in range(k - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, k):
            s -= L[j, i] * b[j]
        b[i] = s / L[i, i]

@njit(cache=True)
def chol_solve_inplace(L, b, k):
    """Overwrite b[:k] with x solving (L L.T) x = b."""
    forward_solve_inplace(L, b, k)
    backward_solve_inplace(L, b, k)

@njit(cache=False, fastmath=True)
def chol_solve_vec_inplace(L, alpha, k):
    """
    alpha: 1D vector, overwritten in-place on alpha[0:k].
    Returns:
      0 if LAPACK used successfully
      1 if fallback used

    LAPACK will be used only if:
      - LAPACK_CHO_SOLVE_ENABLED is True
      - L looks F-contiguous (layout-safe for dpotrs)
      - alpha is contiguous 1D (so reshape(k,1) view is safe)
    """
    if LAPACK_CHO_SOLVE_ENABLED:
        Lt = L.T
        # (k,1) view on the same memory; safe because (k,1) has identical C/F strides
        B = alpha.reshape((k, 1))
        info = _dpotrs_inplace_from_L(Lt, B, k, lower=False)
        if info == 0:
            return 0

    chol_solve_inplace(L, alpha, k)
    return 1

@njit(cache=False, fastmath=True)
def chol_inv_eye(A, L, k):
    """
    Compute inv(A[:k,:k]) using precomputed Cholesky factor L.

    LAPACK path is allowed if L is F-contiguous; identity is symmetric so
    C/F layout is not a correctness issue here, but we still require L F-order.

    Returns:
      invA : (k,k) float64 array
    """
    if LAPACK_CHO_SOLVE_ENABLED:
        Lt = L.T
        Inv = np.eye(k, dtype=np.float64).T
        info = _dpotrs_inplace_from_L(Lt, Inv, k, lower=False)
        if info == 0:
            return Inv

    return np.linalg.solve(A, np.eye(k, dtype=np.float64))

