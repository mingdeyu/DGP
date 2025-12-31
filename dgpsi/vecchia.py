from numba import njit, prange, config, set_num_threads
import numpy as np
from numpy.random import randn
from math import sqrt, exp, fabs, log
from .functions import _Jij_sexp_scaled, _sexp_kvec_invlen, _compute_denoms_from_vs_scaled, _Jii_matern25, _Jij_matern25, _matern25_I1d, _matern25_kvec_invlen, K_matrix_nb, K_sexp_nb, K_matern25_nb
from .chol_backend import forward_solve_inplace, chol_solve_vec_inplace, chol_inv_eye
from scipy.sparse import csr_matrix
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    FAISS_AVAILABLE = False
from psutil import cpu_count

core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num, max_threads)
config.THREADING_LAYER = 'tbb'
set_num_threads(core_num)

SQRT5 = 2.2360679774997898
FIVE_THIRDS = 1.6666666666666667  # 5/3

def get_pred_nn(query, x, m=50, method='exact', size=40, efSearch=100, n_jobs=-1, cast_int32=False):
    n, d = x.shape
    m = min(m, n)

    if m == n:
        k = query.shape[0]
        NN = (np.arange(m, dtype=np.intp) + np.arange(k, dtype=np.intp)[:, None]) % m
        return NN.astype(np.int32, copy=False) if cast_int32 else NN

    if FAISS_AVAILABLE:
        x_in = x
        if x_in.dtype != np.float32 or not x_in.flags['C_CONTIGUOUS']:
            x_in = np.ascontiguousarray(x_in, dtype=np.float32)

        q_in = query
        if q_in.dtype != np.float32 or not q_in.flags['C_CONTIGUOUS']:
            q_in = np.ascontiguousarray(q_in, dtype=np.float32)

        if method == 'exact':
            index = faiss.IndexFlatL2(d)
        elif method == 'approx':
            index = faiss.IndexHNSWFlat(d, size)
            index.hnsw.efSearch = efSearch
        else:
            raise ValueError("method must be 'exact' or 'approx'")

        index.add(x_in)
        _, NN = index.search(q_in, k=int(m))
        return NN.astype(np.int32) if cast_int32 else NN

    # sklearn
    neigh = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)
    neigh.fit(x)
    NN = neigh.kneighbors(query, n_neighbors=m, return_distance=False)
    return NN.astype(np.int32) if cast_int32 else NN


@njit(cache=True)
def nn_brute(x, m):
    n = x.shape[0]
    m = min(m, n - 1)
    NNarray = np.full((n, m + 1), -1, dtype=np.int32)
    for i in range(n):
        dist = np.sum((x[: (i + 1), :] - x[i, :]) ** 2, axis=1)
        order = np.argsort(dist)
        kkeep = min(m + 1, i + 1)
        NNarray[i, :kkeep] = order[:kkeep].astype(np.int32)
    return NNarray

@njit(cache=True)
def fill_vecchia_nn_if_enough(NNarray, query_inds, cand, m_out):
    q, k = cand.shape
    done = np.zeros(q, dtype=np.bool_)
    buf = np.empty(m_out, dtype=np.int32)

    for r in range(q):
        i = query_inds[r]
        t = 0
        for j in range(k):
            v = cand[r, j]
            if v >= 0 and v <= i:
                buf[t] = np.int32(v)
                t += 1
                if t == m_out:
                    for u in range(m_out):
                        NNarray[i, u] = buf[u]
                    done[r] = True
                    break
    return done

def nn(x, m, method="exact", size=40, efSearch=100, n_jobs=-1):
    n, d = x.shape
    m = min(m, n - 1)
    m_out = m + 1

    NNarray = np.full((n, m_out), -1, dtype=np.int32)

    # warm-start
    mult = 2
    maxval = min(mult * m + 1, n)
    NNarray[:maxval] = nn_brute(x[:maxval], m)

    query_inds = np.arange(maxval, n, dtype=np.int32)
    if query_inds.size == 0:
        return np.fliplr(np.sort(NNarray, axis=1))

    # ---- build/search backend ----
    if FAISS_AVAILABLE:
        # faiss wants float32 contiguous; do once
        x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))

        # start with a moderate ksearch
        ksearch = min(n, 2 * m_out)

        while query_inds.size > 0:
            # O(1) prefix bound (query_inds is sorted)
            max_query_inds = int(query_inds[-1]) + 1
            prefix = min(max_query_inds, n)

            if method == "exact":
                index = faiss.IndexFlatL2(d)
            elif method == "approx":
                index = faiss.IndexHNSWFlat(d, size)
                index.hnsw.efSearch = efSearch
            else:
                raise ValueError("method must be 'exact' or 'approx'")
            index.add(x32[:prefix])

            Q = x32[query_inds]
            _, cand = index.search(Q, int(min(ksearch, prefix)))

            done = fill_vecchia_nn_if_enough(NNarray, query_inds, cand, m_out)
            query_inds = query_inds[~done]

            if query_inds.size == 0:
                break

            if ksearch >= prefix:
                for i in query_inds:
                    ii = int(i)
                    dist = np.sum((x[: (ii + 1), :] - x[ii, :]) ** 2, axis=1)
                    order = np.argsort(dist).astype(np.int32)
                    NNarray[ii, :] = order[:m_out]
                break

            ksearch = min(n, 2 * ksearch)

    else:
        neigh = NearestNeighbors(algorithm="kd_tree", n_jobs=n_jobs)

        ksearch = min(n, 2 * m_out)

        while query_inds.size > 0:
            max_query_inds = int(query_inds[-1]) + 1
            prefix = min(max_query_inds, n)

            neigh.fit(x[:prefix, :])
            cand = neigh.kneighbors(
                x[query_inds, :],
                n_neighbors=int(min(ksearch, prefix)),
                return_distance=False,
            )

            done = fill_vecchia_nn_if_enough(NNarray, query_inds, cand, m_out)
            query_inds = query_inds[~done]

            if query_inds.size == 0:
                break

            if ksearch >= prefix:
                for i in query_inds:
                    ii = int(i)
                    dist = np.sum((x[: (ii + 1), :] - x[ii, :]) ** 2, axis=1)
                    order = np.argsort(dist).astype(np.int32)
                    NNarray[ii, :] = order[:m_out]
                break

            ksearch = min(n, 2 * ksearch)

    return np.fliplr(np.sort(NNarray, axis=1))

@njit(cache=True)
def preprocess_nn(NNarray):
    n, m = NNarray.shape
    NN_rev   = np.full((n, m), -1, dtype=NNarray.dtype)
    NN_count = np.empty(n, dtype=np.int32)

    for i in range(n):
        t = 0
        for j in range(m - 1, -1, -1):
            v = NNarray[i, j]
            if v >= 0:
                NN_rev[i, t] = v
                t += 1
        NN_count[i] = t

    return NN_rev, NN_count

@njit(cache=True)
def nn_fwd_from_rev(NN_rev, NN_count):
    """
    Build NN_fwd so that:
      NN_fwd[i,0] is the "self" index (in ordered space),
      and NN_fwd aligns with the columns of Lmat (diag at col 0).
    """
    n, m = NN_rev.shape
    NN_fwd = np.full((n, m), -1, dtype=NN_rev.dtype)
    for i in range(n):
        k = NN_count[i]
        for j in range(k):
            NN_fwd[i, j] = NN_rev[i, k - 1 - j]
    return NN_fwd

@njit(cache=True, fastmath=True)
def forward_solve_sp_inplace(Lmat, NN_fwd, NN_count, x):
    """
    In-place solve: Lmat * x = x  (x initially holds RHS b).
    Uses NN_count to limit loop.
    """
    n = Lmat.shape[0]
    for i in range(n):
        k = NN_count[i]
        s = x[i]
        for j in range(1, k):
            s -= Lmat[i, j] * x[NN_fwd[i, j]]
        x[i] = s / Lmat[i, 0]

@njit(cache=True)
def fmvn_sp(X, NN_rev, NN_count, length, nugget, scale, name):
    """
    X_ord: X already ordered (X[ord])
    returns sample in ordered space
    """
    NN_fwd = nn_fwd_from_rev(NN_rev, NN_count)
    Lmat = L_matrix_nb(X, NN_rev, NN_count, length, nugget, name)
    return fmvn_sp_nb(Lmat, NN_fwd, NN_count, sqrt(float(scale)))

@njit(cache=True)
def fmvn_sp_nb(Lmat, NN_fwd, NN_count, sqrt_scale):
    """
    Draw one sample in *ordered* index space:
      x = L^{-1} (sqrt_scale * z),  z ~ N(0, I)
    Equivalent to your old:
      L = L_matrix(...)/sqrt(scale); solve(L, z)
    but avoids allocating L/sqrt(scale).
    """
    n = Lmat.shape[0]
    x = randn(n).astype(np.float64)
    x *= sqrt_scale
    forward_solve_sp_inplace(Lmat, NN_fwd, NN_count, x)
    return x


@njit(cache=True, fastmath=True)
def forward_solve_last(L, b):
    """Overwrites b with x in Lx=b, returns x[-1]."""
    n = L.shape[0]
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * b[j]
        b[i] = s / L[i, i]
    return b[n - 1]

# for vecchia

@njit(cache=True, fastmath=True)
def solve_lt_e_last_inplace(L, v, k):
    """
    Solve (L.T) v = e_last (e_last[k-1]=1) into v[:k].
    This replaces:
      Ii = zeros(k); Ii[-1]=1; v = solve(L.T, Ii)
    """
    for i in range(k - 1, -1, -1):
        s = 1.0 if i == k - 1 else 0.0
        for j in range(i + 1, k):
            s -= L[j, i] * v[j]          # because (L.T)[i,j] = L[j,i]
        v[i] = s / L[i, i]


# likelihood functions
@njit(cache=True, parallel=True, fastmath=True)
def vecchia_llik(X, y, NN_rev, NN_count, scale, length, nugget, nugget_diag, name):
    n, d = X.shape
    quad, logdet = 0., 0.
    for i in prange(n):
        k = NN_count[i]
        idx_row = NN_rev[i]

        xi = np.empty((k, d), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        for a in range(k):
            ia = idx_row[a]
            yi[a] = y[ia]
            for j in range(d):
                xi[a, j] = X[ia, j]

        Ki = K_matrix_nb(xi, length, 0.0, name)

        for a in range(k):
            Ki[a, a] += nugget * nugget_diag[idx_row[a]]

        Li = np.linalg.cholesky(Ki)

        z_last = forward_solve_last(Li, yi)
        quad += z_last * z_last
        logdet += 2.0 * log(Li[k - 1, k - 1])
    return -0.5 * (logdet + quad / scale) 

@njit(cache=True, parallel=True, fastmath=True)
def vecchia_nllik(
    X, y, NN_rev, NN_count,
    scale, length, nugget, nugget_diag, name,
    scale_est, nugget_est, origin_n, rr
):
    n, d = X.shape
    p = len(length) + (1 if nugget_est else 0)

    quad = 0.0
    logdet = 0.0
    dquad = np.zeros(p, dtype=np.float64)
    dlogdet = np.zeros(p, dtype=np.float64)

    for i in prange(n):
        k = NN_count[i]
        idx_row = NN_rev[i]

        # ---- gather local block (no fancy indexing) ----
        xi = np.empty((k, d), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nug_i = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx_row[a]
            yi[a] = y[ia]
            nug_i[a] = nugget * nugget_diag[ia]
            for j in range(d):
                xi[a, j] = X[ia, j]

        # ---- build K and dK ----
        Ki, dKi = dK_matrix_nb(xi, length, nug_i, name, nugget_est)

        L = np.linalg.cholesky(Ki)

        # z = L^{-1} y (in-place on yi)
        forward_solve_inplace(L, yi, k)
        z_last = yi[k - 1]

        quad_i = z_last * z_last
        logdet_i = 2.0 * log(L[k - 1, k - 1])

        # v = (L.T)^{-1} e_last
        v = np.empty(k, dtype=np.float64)
        # rhs = np.empty(k, dtype=np.float64)
        solve_lt_e_last_inplace(L, v, k)

        dquadi = np.empty(p, dtype=np.float64)
        dlogdeti = np.empty(p, dtype=np.float64)

        for t in range(p):
            # rhs = dKi[t] @ v
            rhs = np.dot(dKi[t], v)
            # rhs = L^{-1} rhs
            forward_solve_inplace(L, rhs, k)

            lid_last = rhs[k - 1]
            si = np.dot(yi, rhs)

            dquadi[t] = 2.0 * si * z_last - lid_last * z_last * z_last
            dlogdeti[t] = lid_last

        quad += quad_i
        logdet += logdet_i
        dquad += dquadi
        dlogdet += dlogdeti

    if scale_est:
        if n == origin_n:
            scale_out = quad / n
            nllik = 0.5 * (logdet + n * log(scale_out))
            grad = 0.5 * (dlogdet - dquad / scale_out)
        else:
            scale_out = (quad + rr / nugget) / origin_n
            nllik = 0.5 * (logdet + origin_n * log(scale_out))
            grad = 0.5 * (dlogdet - dquad / scale_out)
            if nugget_est:
                nllik += 0.5 * (origin_n - n) * log(nugget)
                grad[p - 1] += 0.5 * (-rr / (scale_out * nugget) + (origin_n - n))
    else:
        scale_out = scale
        nllik = 0.5 * (logdet + quad / scale_out)
        grad = 0.5 * (dlogdet - dquad / scale_out)
        if n != origin_n and nugget_est:
            nllik += 0.5 * (rr / (nugget * scale_out) + (origin_n - n) * log(nugget))
            grad[p - 1] += 0.5 * (-rr / (scale_out * nugget) + (origin_n - n))

    return nllik, grad, np.array([scale_out], dtype=np.float64)

@njit(cache=True)
def dK_sexp_nb(xi, length, nugget, nugget_est):
    n, d = xi.shape
    iso = (length.size == 1)

    p_len = 1 if iso else d
    p = p_len + (1 if nugget_est else 0)
    nug_idx = p_len

    K = np.empty((n, n), dtype=np.float64)
    Kt = np.zeros((p, n, n), dtype=np.float64)

    for i in range(n):
        ng = nugget[i]
        K[i, i] = 1.0 + ng
        if nugget_est:
            Kt[nug_idx, i, i] = ng

    if iso:
        inv = 1.0 / length[0]
        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv
                    dist += diff * diff
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
                val = 2.0 * dist * kij
                Kt[0, i, j] = val
                Kt[0, j, i] = val
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        buf = np.empty(d, dtype=np.float64)
        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv_ell[k]
                    t = diff * diff
                    dist += t
                    buf[k] = 2.0 * t
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
                for k in range(d):
                    val = buf[k] * kij
                    Kt[k, i, j] = val
                    Kt[k, j, i] = val
    return K, Kt

@njit(cache=True)
def dK_matern25_nb(xi, length, nugget, nugget_est):
    n, d = xi.shape
    iso = (length.size == 1)

    p_len = 1 if iso else d
    p = p_len + (1 if nugget_est else 0)
    nug_idx = p_len

    K = np.empty((n, n), dtype=np.float64)
    Kt = np.zeros((p, n, n), dtype=np.float64)

    #SQRT5 = 2.2360679774997898
    #FIVE_THIRDS = 1.6666666666666667  # 5/3

    # diagonal
    for i in range(n):
        ng = nugget[i]
        K[i, i] = 1.0 + ng
        if nugget_est:
            Kt[nug_idx, i, i] = ng

    if iso:
        inv = 1.0 / length[0]
        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                coef3 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv)
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                    coef3 += (el2 * el1) / coef

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij

                val = coef3 * kij
                Kt[0, i, j] = val
                Kt[0, j, i] = val
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        buf = np.empty(d, dtype=np.float64)
        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv_ell[k])
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                    buf[k] = (el2 * el1) / coef

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij

                for k in range(d):
                    val = buf[k] * kij
                    Kt[k, i, j] = val
                    Kt[k, j, i] = val
    return K, Kt

@njit(cache=True, parallel=True)
def dK_sexp_nb_parallel(xi, length, nugget, nugget_est):
    n, d = xi.shape
    iso = (length.size == 1)

    p_len = 1 if iso else d
    p = p_len + (1 if nugget_est else 0)
    nug_idx = p_len

    K = np.empty((n, n), dtype=np.float64)
    Kt = np.zeros((p, n, n), dtype=np.float64)

    for i in range(n):
        ng = nugget[i]
        K[i, i] = 1.0 + ng
        if nugget_est:
            Kt[nug_idx, i, i] = ng

    if iso:
        inv = 1.0 / length[0]
        for i in prange(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv
                    dist += diff * diff
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
                val = 2.0 * dist * kij
                Kt[0, i, j] = val
                Kt[0, j, i] = val
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        for i in prange(1, n):
            buf = np.empty(d, dtype=np.float64)
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv_ell[k]
                    t = diff * diff
                    dist += t
                    buf[k] = 2.0 * t
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
                for k in range(d):
                    val = buf[k] * kij
                    Kt[k, i, j] = val
                    Kt[k, j, i] = val
    return K, Kt

@njit(cache=True, parallel=True)
def dK_matern25_nb_parallel(xi, length, nugget, nugget_est):
    n, d = xi.shape
    iso = (length.size == 1)

    p_len = 1 if iso else d
    p = p_len + (1 if nugget_est else 0)
    nug_idx = p_len

    K = np.empty((n, n), dtype=np.float64)
    Kt = np.zeros((p, n, n), dtype=np.float64)

    #SQRT5 = 2.2360679774997898
    #FIVE_THIRDS = 1.6666666666666667  # 5/3

    # diagonal
    for i in range(n):
        ng = nugget[i]
        K[i, i] = 1.0 + ng
        if nugget_est:
            Kt[nug_idx, i, i] = ng

    if iso:
        inv = 1.0 / length[0]
        for i in prange(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                coef3 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv)
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                    coef3 += (el2 * el1) / coef

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij

                val = coef3 * kij
                Kt[0, i, j] = val
                Kt[0, j, i] = val
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        for i in prange(1, n):
            buf = np.empty(d, dtype=np.float64)
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv_ell[k])
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                    buf[k] = (el2 * el1) / coef

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij

                for k in range(d):
                    val = buf[k] * kij
                    Kt[k, i, j] = val
                    Kt[k, j, i] = val
    return K, Kt

@njit(cache=True)
def dK_matrix_nb(xi, length, nugget, kernel_name, nugget_est, parallel = False):
    if kernel_name == "sexp":
        if parallel:
            return dK_sexp_nb_parallel(xi, length, nugget, nugget_est)
        else:
            return dK_sexp_nb(xi, length, nugget, nugget_est)
    else:
        if parallel:
            return dK_matern25_nb_parallel(xi, length, nugget, nugget_est)
        else:
            return dK_matern25_nb(xi, length, nugget, nugget_est)

@njit(cache=True, parallel=True)
def L_matrix_nb(X_ord, NN_rev, NN_count, length, nugget, name):
    """
    Build sparse Vecchia factor coefficients Lmat (n,m):
      - Lmat[i,0] is the diagonal coefficient for x[i]
      - Lmat[i,1:k] align with neighbor indices NN_fwd[i,1:k]
    X_ord is already ordered (X[ord]).
    """
    n, d = X_ord.shape
    m = NN_rev.shape[1]
    Lmat = np.zeros((n, m), dtype=np.float64)

    for i in prange(n):
        k = NN_count[i]
        idx_rev = NN_rev[i]

        # gather xi in idx_rev order (self is last in this order)
        xi = np.empty((k, d), dtype=np.float64)
        for a in range(k):
            ia = idx_rev[a]
            for j in range(d):
                xi[a, j] = X_ord[ia, j]

        Ki = K_matrix_nb(xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)

        # v = (Li.T)^{-1} e_last
        v = np.empty(k, dtype=np.float64)
        solve_lt_e_last_inplace(Li, v, k)

        # store in forward order (reverse v): diag first
        for j in range(k):
            Lmat[i, j] = v[k - 1 - j]

        # rest stays 0
    return Lmat

@njit(cache=True, parallel=True)
def U_matrix_ptr(X, NNarray, n_obs, length, nugget, scale, gamma, name):
    n, m = NNarray.shape
    d = X.shape[1]

    data = np.empty(n * m, dtype=np.float64)

    for i in prange(n):
        # Build idx in the same order as revNNarray row: NNarray[i, m-1], ..., NNarray[i, 0]
        idx_mod = np.empty(m, dtype=np.int32)     
        diag_add = np.empty(m, dtype=np.float64)  

        t = 0
        for j in range(m - 1, -1, -1):            
            idx = NNarray[i, j]
            cond = idx >= n_obs                   
            ia = idx - n_obs if cond else idx     

            idx_mod[t] = ia
            diag_add[t] = (0.0 if cond else gamma[ia]) + 1e-10
            t += 1

        xi = X[idx_mod, :]                       

        Ki = scale * K_matrix_nb(xi, length, nugget, name) 
        add_to_diag_square(Ki, diag_add)                  
        Li = np.linalg.cholesky(Ki)                        

        v = np.empty(m, dtype=np.float64)                   
        solve_lt_e_last_inplace(Li, v, m)                         

        off = i * m
        for j in range(m):
            data[off + j] = v[j]
    return data

@njit(cache=True)
def imp_pointers(NNarray):
    n, m = NNarray.shape
    rowp = np.empty(n * m, dtype=np.int32)
    colp = np.empty(n * m, dtype=np.int32)

    cur = 0
    for i in range(n):
        for j in range(m - 1, -1, -1):
            v = NNarray[i, j]
            if v >= 0:
                rowp[cur] = i
                colp[cur] = v
                cur += 1
    return rowp, colp

@njit(cache=True)
def add_to_diag_square(A, d):
    n = A.shape[0]
    flat = A.ravel()
    step = n + 1
    flat[0 : n*n : step] += d[:n]

def U_matrix_sp_nb(X, NNarray, scale, length, nugget, name, gamma, rows, cols):
    n = X.shape[0]

    data = U_matrix_ptr(X, NNarray, n, length, nugget, scale, gamma, name)

    U = csr_matrix((data, (cols, rows)), shape=(2 * n, n))
    U_latent = U[n::, :]
    U_obs_latent = U[:n, :]
    return U_latent, U_obs_latent


def cond_mean_vecch(x, z, w1, global_w1, y, scale, length, nugget, name, m, nn_method):
    """Make GP mean predictions with Vecchia approximation in initialisation.
    """
    if z is not None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    NNarray = get_pred_nn(x/length, w1/length, m, method = nn_method)
    nugget_diag = np.ones(len(y))
    m,_ = gp_vecch(x, w1, NNarray, y, scale[0], length, nugget[0], nugget_diag, name)
    return m

@njit(cache=True, parallel=True)
def gp_vecch(x,w,NNarray,y,scale,length,nugget,nugget_diag,name):
    """Make GP predictions with Vecchia approximation.
    """
    n_pred, d = x.shape
    k = NNarray.shape[1]
    m_out = np.empty(n_pred)
    v_out = np.empty(n_pred)

    for i in prange(n_pred):
        idx = NNarray[i]    
        Xi = np.empty((k + 1, d), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            for j in range(d):
                Xi[a, j] = w[ia, j]
        for j in range(d):
            Xi[k, j] = x[i, j]

        Ki = K_matrix_nb(Xi, length, 0.0, name)

        for a in range(k):
            ia = idx[a]
            Ki[a, a] += nugget * nugget_diag[ia]
        Ki[k, k] += nugget

        Li = np.linalg.cholesky(Ki)
        forward_solve_inplace(Li, yi, k)
        s = 0.0
        for j in range(k):
            s += Li[k, j] * yi[j]
        m_out[i] = s

        t = Li[k, k]
        v_out[i] = scale * (t * t)
    return m_out, v_out


@njit(cache=True, parallel=True)
def loo_gp_vecch(x, NNarray, y, scale, length, nugget, nugget_diag, name):
    """Compute LOO for GP with Vecchia approximation."""
    n_pred, d = x.shape
    k = NNarray.shape[1]

    m_out = np.empty(n_pred, dtype=np.float64)
    v_out = np.empty(n_pred, dtype=np.float64)

    for i in prange(n_pred):
        idx_row = NNarray[i]

        # Fill in reverse to match idx[::-1]
        Xi = np.empty((k, d), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nuggeti = np.empty(k, dtype=np.float64)

        t = 0
        for a in range(k - 1, -1, -1):
            ia = idx_row[a]
            yi[t] = y[ia, 0]
            nuggeti[t] = nugget * nugget_diag[ia]
            for j in range(d):
                Xi[t, j] = x[ia, j]
            t += 1

        Ki = K_matrix_nb(Xi, length, nuggeti, name)
        Li = np.linalg.cholesky(Ki)

        # mean = Li[-1,:-1] @ (L11^{-1} yi[:-1])  (solve in-place on yi)
        forward_solve_inplace(Li, yi, k - 1)
        s = 0.0
        for j in range(k - 1):
            s += Li[k - 1, j] * yi[j]

        m_out[i] = s
        tdiag = Li[k - 1, k - 1]
        v_out[i] = scale * (tdiag * tdiag)

    return m_out, v_out

@njit(cache=True)
def gp_vecch_non_parallel(x, w, NNarray, y, scale, length, nugget, nugget_diag, name):
    """Make GP predictions with Vecchia approximation."""
    n_pred, d = x.shape
    k = NNarray.shape[1]
    m_out = np.empty(n_pred)
    v_out = np.empty(n_pred)

    for i in range(n_pred):
        idx = NNarray[i]
        Xi = np.empty((k + 1, d), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            for j in range(d):
                Xi[a, j] = w[ia, j]
        for j in range(d):
            Xi[k, j] = x[i, j]

        Ki = K_matrix_nb(Xi, length, 0.0, name)

        for a in range(k):                      
            ia = idx[a]
            Ki[a, a] += nugget * nugget_diag[ia]
        Ki[k, k] += nugget

        Li = np.linalg.cholesky(Ki)
        forward_solve_inplace(Li, yi, k)       
        s = 0.0                              
        for j in range(k):
            s += Li[k, j] * yi[j]
        m_out[i] = s

        t = Li[k, k]
        v_out[i] = scale * (t * t)        
    return m_out, v_out

@njit(cache=True)
def forward_substitute(L_data, L_indices, L_indptr, b):
    """
    Solves Lx = b for x, where L is a lower triangular matrix in CSR format.
    - L_data: Non-zero values of L.
    - L_indices: Column indices of non-zero values in L.
    - L_indptr: Index pointers for rows in L.
    - b: Right-hand side vector.
    Returns:
    - x: Solution vector.
    """
    n = len(L_indptr) - 1  # Number of rows
    x = np.empty(n)  # Solution vector
    for i in range(n):
        sum_lx = 0.
        for j in range(L_indptr[i], L_indptr[i + 1]):
            col = L_indices[j]
            if col < i:
                sum_lx += L_data[j] * x[col]
            elif col == i:
                x[i] = (b[i] - sum_lx) / L_data[j]
                break  # Only one diagonal element per row, so can break after processing it
    return x

# linked gp prediction (sexp, vecchia)

@njit(cache=False, fastmath=True)
def link_gp_vecch_sexp_noz_serial(m, v, w1, NNarray, y,
                                 scale, length_w, inv_len_w,
                                 nugget, nugget_diag):
    n_pred, d = m.shape
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    k = NNarray.shape[1]

    for t in range(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, w1.shape[1]), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64) 
        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(w1.shape[1]):
                wi[a, j] = w1[ia, j]

        Ki = K_sexp_nb(wi, length_w, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi

        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        denI = np.empty(d, dtype=np.float64)
        denJ = np.empty(d, dtype=np.float64)

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for kk in range(d):
            inv = inv_len_w[kk]
            ms[kk] = m[t, kk] * inv
            vs[kk] = v[t, kk] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            expo = 0.0
            for kk in range(d):
                xi = wi[i, kk] * inv_len_w[kk]
                diff = xi - ms[kk]
                expo += diff * diff * denI[kk]
            Ii = I_coef1 * exp(-expo)

            ai = alpha[i]
            IR += Ii * ai

            Jii = _Jij_sexp_scaled(wi, i, i, ms, denJ, inv_len_w, J_coef1)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                Jij = _Jij_sexp_scaled(wi, i, j, ms, denJ, inv_len_w, J_coef1)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

@njit(cache=False, parallel=True, fastmath=True)
def link_gp_vecch_sexp_noz_parallel(m, v, w1, NNarray, y,
                                   scale, length_w, inv_len_w,
                                   nugget, nugget_diag):
    n_pred, d = m.shape
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    k = NNarray.shape[1]

    for t in prange(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, w1.shape[1]), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64) 
        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(w1.shape[1]):
                wi[a, j] = w1[ia, j]

        Ki = K_sexp_nb(wi, length_w, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        denI = np.empty(d, dtype=np.float64)
        denJ = np.empty(d, dtype=np.float64)

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for kk in range(d):
            inv = inv_len_w[kk]
            ms[kk] = m[t, kk] * inv
            vs[kk] = v[t, kk] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            expo = 0.0
            for kk in range(d):
                xi = wi[i, kk] * inv_len_w[kk]
                diff = xi - ms[kk]
                expo += diff * diff * denI[kk]
            Ii = I_coef1 * exp(-expo)

            ai = alpha[i]
            IR += Ii * ai

            Jii = _Jij_sexp_scaled(wi, i, i, ms, denJ, inv_len_w, J_coef1)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                Jij = _Jij_sexp_scaled(wi, i, j, ms, denJ, inv_len_w, J_coef1)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

# ============================================================
# SEXP: z is not None
# ============================================================

@njit(cache=False, fastmath=True)
def link_gp_vecch_sexp_withz_serial(m, v, z, w1, global_w1, NNarray, y,
                                   scale, length_full, inv_len_w, inv_len_z,
                                   nugget, nugget_diag):
    n_pred, d = m.shape
    Dz = z.shape[1]
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    for t in range(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        gi = np.empty((k, Dz), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        xfull = np.empty((k, Dw + Dz), dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]

            for j in range(Dw):
                val = w1[ia, j]
                wi[a, j] = val
                xfull[a, j] = val

            for j in range(Dz):
                val = global_w1[ia, j]
                gi[a, j] = val
                xfull[a, Dw + j] = val

        Ki = K_sexp_nb(xfull, length_full, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi

        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        Izi = _sexp_kvec_invlen(gi, z[t], inv_len_z)

        denI = np.empty(d, dtype=np.float64)
        denJ = np.empty(d, dtype=np.float64)

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for kk in range(d):
            inv = inv_len_w[kk]
            ms[kk] = m[t, kk] * inv
            vs[kk] = v[t, kk] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            expo = 0.0
            for kk in range(d):
                xi = wi[i, kk] * inv_len_w[kk]
                diff = xi - ms[kk]
                expo += diff * diff * denI[kk]
            Ii = I_coef1 * exp(-expo)

            si = Izi[i]
            ai = alpha[i]
            IR += (Ii * si) * ai

            Jii = _Jij_sexp_scaled(wi, i, i, ms, denJ, inv_len_w, J_coef1) * (si * si)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                sj = Izi[j]
                Jij = _Jij_sexp_scaled(wi, i, j, ms, denJ, inv_len_w, J_coef1) * (si * sj)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

@njit(cache=False, parallel=True, fastmath=True)
def link_gp_vecch_sexp_withz_parallel(m, v, z, w1, global_w1, NNarray, y,
                                     scale, length_full, inv_len_w, inv_len_z,
                                     nugget, nugget_diag):
    n_pred, d = m.shape
    Dz = z.shape[1]
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    for t in prange(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        gi = np.empty((k, Dz), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        xfull = np.empty((k, Dw + Dz), dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]

            for j in range(Dw):
                val = w1[ia, j]
                wi[a, j] = val
                xfull[a, j] = val

            for j in range(Dz):
                val = global_w1[ia, j]
                gi[a, j] = val
                xfull[a, Dw + j] = val

        Ki = K_sexp_nb(xfull, length_full, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        Izi = _sexp_kvec_invlen(gi, z[t], inv_len_z)

        denI = np.empty(d, dtype=np.float64)
        denJ = np.empty(d, dtype=np.float64)

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for kk in range(d):
            inv = inv_len_w[kk]
            ms[kk] = m[t, kk] * inv
            vs[kk] = v[t, kk] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            expo = 0.0
            for kk in range(d):
                xi = wi[i, kk] * inv_len_w[kk]
                diff = xi - ms[kk]
                expo += diff * diff * denI[kk]
            Ii = I_coef1 * exp(-expo)

            si = Izi[i]
            ai = alpha[i]
            IR += (Ii * si) * ai

            Jii = _Jij_sexp_scaled(wi, i, i, ms, denJ, inv_len_w, J_coef1) * (si * si)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                sj = Izi[j]
                Jij = _Jij_sexp_scaled(wi, i, j, ms, denJ, inv_len_w, J_coef1) * (si * sj)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

# ============================================================
# MATERN2.5: z is None
# ============================================================

@njit(cache=False, fastmath=True)
def link_gp_vecch_matern25_noz_serial(m, v, w1, NNarray, y, scale, length_w, nugget, nugget_diag):

    n_pred, d = m.shape
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in range(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)
        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(Dw):
                wi[a, j] = w1[ia, j]

        Ki = K_matern25_nb(wi, length_w, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            Ii = 1.0
            for kk in range(d):
                Ii *= _matern25_I1d(wi[i, kk], m_row[kk], v_row[kk],
                                    length_w[kk], inv_len_w[kk], inv_len2_w[kk])

            ai = alpha[i]
            IR += Ii * ai

            Jii = _Jii_matern25(wi, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                Jij = _Jij_matern25(wi, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=False, parallel=True, fastmath=True)
def link_gp_vecch_matern25_noz_parallel(m, v, w1, NNarray, y, scale, length_w, nugget, nugget_diag):
    n_pred, d = m.shape
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in prange(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)
        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(Dw):
                wi[a, j] = w1[ia, j]

        Ki = K_matern25_nb(wi, length_w, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            Ii = 1.0
            for kk in range(d):
                Ii *= _matern25_I1d(wi[i, kk], m_row[kk], v_row[kk],
                                    length_w[kk], inv_len_w[kk], inv_len2_w[kk])

            ai = alpha[i]
            IR += Ii * ai

            Jii = _Jii_matern25(wi, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                Jij = _Jij_matern25(wi, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

# ============================================================
# MATERN2.5: z is not None
# ============================================================

@njit(cache=False, fastmath=True)
def link_gp_vecch_matern25_withz_serial(m, v, z, w1, global_w1, NNarray, y, scale, length_full, length_w, inv_len_z, nugget, nugget_diag):
    n_pred, d = m.shape
    Dz = z.shape[1]
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in range(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        gi = np.empty((k, Dz), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        xfull = np.empty((k, Dw + Dz), dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(Dw):
                val = w1[ia, j]
                wi[a, j] = val
                xfull[a, j] = val

            for j in range(Dz):
                val = global_w1[ia, j]
                gi[a, j] = val
                xfull[a, Dw + j] = val

        Ki = K_matern25_nb(xfull, length_full, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        Izi = _matern25_kvec_invlen(gi, z[t], inv_len_z)

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            Ii = 1.0
            for kk in range(d):
                Ii *= _matern25_I1d(wi[i, kk], m_row[kk], v_row[kk],
                                    length_w[kk], inv_len_w[kk], inv_len2_w[kk])

            si = Izi[i]
            ai = alpha[i]
            IR += (Ii * si) * ai

            Jii = _Jii_matern25(wi, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w) * (si * si)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                sj = Izi[j]
                Jij = _Jij_matern25(wi, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w) * (si * sj)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=False, parallel=True, fastmath=True)
def link_gp_vecch_matern25_withz_parallel(m, v, z, w1, global_w1, NNarray, y, scale, length_full, length_w, inv_len_z, nugget, nugget_diag):
    n_pred, d = m.shape
    Dz = z.shape[1]
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in prange(n_pred):
        idx = NNarray[t]

        wi = np.empty((k, Dw), dtype=np.float64)
        gi = np.empty((k, Dz), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        xfull = np.empty((k, Dw + Dz), dtype=np.float64)
        nug0 = np.empty(k, dtype=np.float64)

        for a in range(k):
            ia = idx[a]
            yi[a] = y[ia, 0]
            nug0[a] = nugget * nugget_diag[ia]
            for j in range(Dw):
                val = w1[ia, j]
                wi[a, j] = val
                xfull[a, j] = val

            for j in range(Dz):
                val = global_w1[ia, j]
                gi[a, j] = val
                xfull[a, Dw + j] = val

        Ki = K_matern25_nb(xfull, length_full, nug0)
        L = np.linalg.cholesky(Ki)

        alpha = yi
        #chol_solve_inplace(L, alpha, k)
        chol_solve_vec_inplace(L, alpha, k)

        #Kinv = np.linalg.solve(Ki, np.eye(k))
        Kinv = chol_inv_eye(Ki, L, k)

        Izi = _matern25_kvec_invlen(gi, z[t], inv_len_z)

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(k):
            Ii = 1.0
            for kk in range(d):
                Ii *= _matern25_I1d(wi[i, kk], m_row[kk], v_row[kk],
                                    length_w[kk], inv_len_w[kk], inv_len2_w[kk])

            si = Izi[i]
            ai = alpha[i]
            IR += (Ii * si) * ai

            Jii = _Jii_matern25(wi, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w) * (si * si)
            tr += Kinv[i, i] * Jii
            quad += Jii * ai * ai

            for j in range(i):
                sj = Izi[j]
                Jij = _Jij_matern25(wi, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w) * (si * sj)
                tr += 2.0 * Kinv[i, j] * Jij
                quad += 2.0 * Jij * ai * alpha[j]

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new
