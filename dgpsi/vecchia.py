from numba import njit, prange, config, vectorize, float64, set_num_threads
import numpy as np
from numpy.random import randn
from math import erf, sqrt, pi, exp, fabs, log
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

def _build_knn_index(x, method='exact', size=40, efSearch=100, n_jobs=-1):
    """
    Build a kNN index on x.

    Returns:
      index: FAISS index or sklearn NearestNeighbors instance
      x_index: only for FAISS: float32 contiguous version of x actually stored in the index
               for sklearn: None
    """
    n, d = x.shape

    if FAISS_AVAILABLE:
        if method == 'exact':
            index = faiss.IndexFlatL2(d)
        elif method == 'approx':
            index = faiss.IndexHNSWFlat(d, size)
            index.hnsw.efSearch = efSearch
        else:
            raise ValueError("method must be 'exact' or 'approx'")

        index.add(x)
        return index

    index = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)
    index.fit(x)
    return index

def get_pred_nn(query, x, m=50, method='exact', size=40, efSearch=100, n_jobs=-1):
    n, d = x.shape
    m = min(m, n)

    if m == n:
        k = query.shape[0]
        NN = np.arange(m, dtype=np.int32) + np.arange(k, dtype=np.int32)[:, None]
        NN %= m
        return NN

    index = _build_knn_index(x, method=method, size=size, efSearch=efSearch, n_jobs=n_jobs)

    if FAISS_AVAILABLE:
        _, NN = index.search(query, int(m))
        return NN.astype(np.int32, copy=False)

    NN = index.kneighbors(query, n_neighbors=m, return_distance=False)
    return NN.astype(np.int32, copy=False)

@njit(cache=True)
def nn_brute(x, m):
    n = x.shape[0]
    m = min(m, n - 1)
    NNarray = np.full((n, m + 1), -1, dtype=np.int32)
    for i in range(n):
        # dist to prefix 0..i
        dist = np.sum((x[:(i + 1), :] - x[i, :]) ** 2, axis=1)
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
                buf[t] = v
                t += 1
                if t == m_out:
                    # commit only when complete
                    for u in range(m_out):
                        NNarray[i, u] = buf[u]
                    done[r] = True
                    break
    return done

def nn(x, m, method='exact', size=40, efSearch=100, n_jobs=-1):
    """
    Compute Vecchia NNarray (n, m+1) with the constraint neighbors must be <= row index.
    """
    n, d = x.shape
    if n == 0:
        return np.empty((0, 0), dtype=np.int32)

    m = min(m, n - 1)
    NNarray = np.full((n, m + 1), -1, dtype=np.int32)

    # brute warm-start
    mult = 2
    maxval = min(mult * m + 1, n)
    NNarray[:maxval] = nn_brute(x[:maxval], m)

    if maxval >= n:
        return np.fliplr(np.sort(NNarray, axis=1))

    index = _build_knn_index(x, method=method, size=size, efSearch=efSearch, n_jobs=n_jobs)

    query_inds = np.arange(maxval, n, dtype=np.int32)
    ksearch = min(n, 2 * (m + 1))

    while query_inds.size > 0:
        if FAISS_AVAILABLE:
            # query from float32 stored matrix
            Q = x[query_inds]
            _, cand = index.search(Q, int(ksearch))
        else:
            Q = x[query_inds]
            cand = index.kneighbors(Q, n_neighbors=int(ksearch), return_distance=False)

        cand = cand.astype(np.int32, copy=False)

        done = fill_vecchia_nn_if_enough(NNarray, query_inds, cand, m + 1)
        query_inds = query_inds[~done]

        if query_inds.size == 0:
            break

        if ksearch >= n:
            for i in query_inds:
                dist = np.sum((x[:(i + 1), :] - x[i, :]) ** 2, axis=1)
                order = np.argsort(dist).astype(np.int32)
                NNarray[i, :] = order[:(m + 1)]
            break

        ksearch = min(n, 2 * ksearch)

    NNarray = np.fliplr(np.sort(NNarray, axis=1))
    return NNarray

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

@njit(cache=True)
def forward_solve(L, b):
    n = L.shape[0]
    x = np.zeros((n,1))
    for i in range(n):
        sumj = 0.0
        for j in range(i):
            sumj += L[i, j] * x[j,0]
        x[i] = (b[i] - sumj) / L[i, i]
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
def forward_solve_inplace(L, b, k):
    """Overwrite b[:k] with x solving L x = b (L lower-triangular)."""
    for i in range(k):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * b[j]
        b[i] = s / L[i, i]

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

@njit(cache=True, fastmath=True)
def dot_k(a, b, k):
    s = 0.0
    for i in range(k):
        s += a[i] * b[i]
    return s


@njit(cache=True, fastmath=True)
def matvec_k(A, x, out, k):
    for i in range(k):
        s = 0.0
        for j in range(k):
            s += A[i, j] * x[j]
        out[i] = s

@njit(cache=True)
def backward_solve(U, b):
    n = U.shape[0]
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        sumj = 0.0
        for j in range(i+1, n):
            sumj += U[i, j] * x[j,0]
        x[i] = (b[i] - sumj) / U[i, i]
    return x

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
        rhs = np.empty(k, dtype=np.float64)
        solve_lt_e_last_inplace(L, v, k)

        dquadi = np.empty(p, dtype=np.float64)
        dlogdeti = np.empty(p, dtype=np.float64)

        for t in range(p):
            # rhs = dKi[t] @ v
            matvec_k(dKi[t], v, rhs, k)
            # rhs = L^{-1} rhs
            forward_solve_inplace(L, rhs, k)

            lid_last = rhs[k - 1]
            si = dot_k(yi, rhs, k)

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
def K_vec_nb(X, z, name):
    """Compute cross-correlation vector between training inputs X and a single test point z."""
    if name == "sexp":                                           
        return K_vec_sexp_nb(X, z)                      
    else:
        return K_vec_matern25_nb(X, z)                


@njit(cache=True, fastmath=True)
def K_vec_sexp_nb(X, z):
    n1, d = X.shape                                  
    K_vec = np.empty(n1, dtype=np.float64)   
    for i in range(n1):
        Xi = X[i]
        dist = 0.0
        for k in range(d):
            diff = Xi[k] - z[k]                  
            dist += diff * diff
        K_vec[i] = exp(-dist)
    return K_vec                    

@njit(cache=True, fastmath=True)
def K_vec_matern25_nb(X, z):
    n1, d = X.shape
    K_vec = np.empty(n1, dtype=np.float64)                    

    SQRT5 = 2.2360679774997898                                   
    FIVE_THIRDS = 1.6666666666666667

    for i in range(n1):
        Xi = X[i]
        coef1 = 1.0
        coef2 = 0.0
        for k in range(d):
            distk = fabs(Xi[k] - z[k])
            coef1 *= (1.0 + SQRT5 * distk + FIVE_THIRDS * distk * distk)
            coef2 += distk
        K_vec[i] = coef1 * exp(-SQRT5 * coef2)
    return K_vec

@njit(cache=True)
def K_sexp_nb(xi, length, nugget):
    n, d = xi.shape
    iso = (length.size == 1)

    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = (nugget.size == 1)

    if nug_scalar:
        ng0 = nugget[0]
        for i in range(n):
            K[i, i] = 1.0 + ng0
    else:
        for i in range(n):
            K[i, i] = 1.0 + nugget[i]

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
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv_ell[k]
                    t = diff * diff
                    dist += t
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
    return K

@njit(cache=True)
def K_matern25_nb(xi, length, nugget):
    n, d = xi.shape
    iso = (length.size == 1)

    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = (nugget.size == 1)

    SQRT5 = 2.2360679774997898
    FIVE_THIRDS = 1.6666666666666667  # 5/3

    # diagonal
    if nug_scalar:
        ng0 = nugget[0]
        for i in range(n):
            K[i, i] = 1.0 + ng0
    else:
        for i in range(n):
            K[i, i] = 1.0 + nugget[i]

    if iso:
        inv = 1.0 / length[0]
        for i in range(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv)
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

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

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij
    return K

@njit(cache=True, parallel=True)
def K_sexp_nb_parallel(xi, length, nugget):
    n, d = xi.shape
    iso = (length.size == 1)

    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = (nugget.size == 1)

    if nug_scalar:
        ng0 = nugget[0]
        for i in range(n):
            K[i, i] = 1.0 + ng0
    else:
        for i in range(n):
            K[i, i] = 1.0 + nugget[i]

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
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        for i in prange(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                dist = 0.0
                for k in range(d):
                    diff = (xi_i[k] - xi_j[k]) * inv_ell[k]
                    t = diff * diff
                    dist += t
                kij = exp(-dist)
                K[i, j] = kij
                K[j, i] = kij
    return K

@njit(cache=True, parallel=True)
def K_matern25_nb_parallel(xi, length, nugget):
    n, d = xi.shape
    iso = (length.size == 1)

    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = (nugget.size == 1)

    SQRT5 = 2.2360679774997898
    FIVE_THIRDS = 1.6666666666666667  # 5/3

    # diagonal
    if nug_scalar:
        ng0 = nugget[0]
        for i in range(n):
            K[i, i] = 1.0 + ng0
    else:
        for i in range(n):
            K[i, i] = 1.0 + nugget[i]

    if iso:
        inv = 1.0 / length[0]
        for i in prange(1, n):
            xi_i = xi[i]
            for j in range(i):
                xi_j = xi[j]
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi_i[k] - xi_j[k]) * inv)
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

        for i in prange(1, n):
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

                kij = coef1 * exp(-SQRT5 * coef2)
                K[i, j] = kij
                K[j, i] = kij
    return K

@njit(cache=True)
def K_matrix_nb(xi, length, nugget, name, parallel = False):
    nugget = np.atleast_1d(np.asarray(nugget, dtype=np.float64))
    if name == "sexp":
        if parallel:
            return K_sexp_nb_parallel(xi, length, nugget)
        else:
            return K_sexp_nb(xi, length, nugget)
    else:
        if parallel:
            return K_matern25_nb_parallel(xi, length, nugget)
        else:
            return K_matern25_nb(xi, length, nugget)

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

    SQRT5 = 2.2360679774997898
    FIVE_THIRDS = 1.6666666666666667  # 5/3

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

    SQRT5 = 2.2360679774997898
    FIVE_THIRDS = 1.6666666666666667  # 5/3

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
    m_out = np.zeros(n_pred)
    v_out = np.zeros(n_pred)

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
def loo_gp_vecch(x,NNarray,y,scale,length,nugget,nugget_diag,name):
    """Compute LOO for GP with Vecchia approximation.
    """
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in prange(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        Xi = x[idx,:]
        nuggeti = nugget * nugget_diag[idx]
        Ki = K_matrix_nb(Xi, length, 0., name)
        add_to_diag_square(Ki, nuggeti)
        Li = np.linalg.cholesky(Ki)
        yi = y[idx,0]
        m[i] = np.dot(Li[-1,:-1], forward_solve(Li[:-1, :-1], yi[:-1]).flatten())
        v[i] = scale * Li[-1,-1]**2
    return m, v

@njit(cache=True)
def gp_vecch_non_parallel(x, w, NNarray, y, scale, length, nugget, nugget_diag, name):
    """Make GP predictions with Vecchia approximation."""
    n_pred, d = x.shape
    k = NNarray.shape[1]
    m_out = np.zeros(n_pred)
    v_out = np.zeros(n_pred)

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
    
@njit(cache=True, parallel=True)
def link_gp_vecch(m, v, z, w1, global_w1, NNarray, y, scale, length, nugget, nugget_diag, name):
    """Make linked GP predictions.
    """
    n_pred = m.shape[0]
    m_new, v_new = np.zeros(n_pred), np.zeros(n_pred)
    if z is not None:
        Dw=np.shape(w1)[1]
        Dz=np.shape(z)[1]
        if len(length)==1:
            length=np.full(Dw+Dz, length[0])
    else:
        Dw=np.shape(w1)[1]
        if len(length)==1:
            length=np.full(Dw, length[0])
    for i in prange(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0]
        yi = y[idx,0]
        nuggeti = nugget * nugget_diag[idx]
        if z is not None:
            wi, global_wi = w1[idx,:], global_w1[idx,:]
            Izi = K_vec_nb(global_wi, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            Ii,Ji = IJ_nb(wi, m[i], v[i], length[:-Dz], name)
            Ii,Ji = Ii*Izi, Ji*Jzi
            Ki = K_matrix_nb(np.concatenate((wi, global_wi),1), length, 0., name)
        else:
            wi = w1[idx,:]
            Ii,Ji = IJ_nb(wi, m[i], v[i], length, name)
            Ki = K_matrix_nb(wi, length, 0., name)
        add_to_diag_square(Ki, nuggeti)
        tr_RinvJ=np.trace(np.linalg.solve(Ki,Ji))
        Li = np.linalg.cholesky(Ki)
        Rinv_y = backward_solve(Li.T, forward_solve(Li, yi).flatten()).flatten()
        IRinv_y = np.dot(Ii,Rinv_y)
        m_new[i] = IRinv_y
        v_new[i] = np.abs(quad(Ji,Rinv_y)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new,v_new

@njit(cache=True)
def link_gp_vecch_non_parallel(m, v, z, w1, global_w1, NNarray, y, scale, length, nugget, nugget_diag, name):
    """Make linked GP predictions.
    """
    n_pred = m.shape[0]
    m_new, v_new = np.zeros(n_pred), np.zeros(n_pred)
    if z is not None:
        Dw=np.shape(w1)[1]
        Dz=np.shape(z)[1]
        if len(length)==1:
            length=np.full(Dw+Dz, length[0])
    else:
        Dw=np.shape(w1)[1]
        if len(length)==1:
            length=np.full(Dw, length[0])
    for i in range(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0]
        yi = y[idx,0]
        nuggeti = nugget * nugget_diag[idx]
        if z is not None:
            wi, global_wi = w1[idx,:], global_w1[idx,:]
            Izi = K_vec_nb(global_wi, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            Ii,Ji = IJ_nb(wi, m[i], v[i], length[:-Dz], name)
            Ii,Ji = Ii*Izi, Ji*Jzi
            Ki = K_matrix_nb(np.concatenate((wi, global_wi),1), length, 0., name)
        else:
            wi = w1[idx,:]
            Ii,Ji = IJ_nb(wi, m[i], v[i], length, name)
            Ki = K_matrix_nb(wi, length, 0., name)
        add_to_diag_square(Ki, nuggeti)
        tr_RinvJ=np.trace(np.linalg.solve(Ki,Ji))
        Li = np.linalg.cholesky(Ki)
        Rinv_y = backward_solve(Li.T, forward_solve(Li, yi).flatten()).flatten()
        IRinv_y = np.dot(Ii,Rinv_y)
        m_new[i] = IRinv_y
        v_new[i] = np.abs(quad(Ji,Rinv_y)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new,v_new

@njit(cache=True)
def IJ_nb(X, z_m, z_v, length, name):
    """Compute I and J involved in linked GP predictions.
    """
    n, d = X.shape
    I = np.zeros(n)
    J = np.zeros((n,n))
    if name == 'sexp':
        X_z = X-z_m
        I_coef1, J_coef1 = 1., 1.
        for k in range(d):
            div = 2*z_v[k]/length[k]**2
            I_coef1 *= 1 + div
            J_coef1 *= 1 + 2*div
        I_coef1, J_coef1 = 1/sqrt(I_coef1), 1/sqrt(J_coef1)
        for i in range(n):
            I_coef2 = 0.
            for k in range(d):
                I_coef2 += X_z[i,k]**2/(2*z_v[k]+length[k]**2)
            I[i] = I_coef1 * np.exp(-I_coef2)
            for j in range( i + 1 ):
                if i==j:
                    J_coef2 = 0.
                    for k in range(d):
                        J_coef2 += 2*X_z[i,k]**2/(4*z_v[k]+length[k]**2)
                    J[i,j] = J_coef1 * np.exp(-J_coef2)
                else:
                    J_coef2 = 0.
                    for k in range(d):
                        J_coef2 += (X_z[i,k] + X_z[j,k])**2/(8*z_v[k]+2*length[k]**2)+(X_z[i,k] - X_z[j,k])**2/(2*length[k]**2)
                    J[i,j] = J_coef1 * np.exp(-J_coef2)
                    J[j,i] = J[i,j]
    elif name=='matern2.5':
        zX = z_m-X
        muA, muB = zX-sqrt(5)*z_v/length, zX+sqrt(5)*z_v/length
        for i in range(n):
            Ii = 1.
            for k in range(d):
                if z_v[k]!=0:
                    Ii *= np.exp((5*z_v[k]-2*sqrt(5)*length[k]*zX[i,k])/(2*length[k]**2))* \
                        ((1+sqrt(5)*muA[i,k]/length[k]+5*(muA[i,k]**2+z_v[k])/(3*length[k]**2))*0.5*(1+erf(muA[i,k]/sqrt(2*z_v[k])))+ \
                        (sqrt(5)+(5*muA[i,k])/(3*length[k]))*sqrt(0.5*z_v[k]/pi)/length[k]*np.exp(-0.5*muA[i,k]**2/z_v[k]))+ \
                        np.exp((5*z_v[k]+2*sqrt(5)*length[k]*zX[i,k])/(2*length[k]**2))* \
                        ((1-sqrt(5)*muB[i,k]/length[k]+5*(muB[i,k]**2+z_v[k])/(3*length[k]**2))*0.5*(1+erf(-muB[i,k]/sqrt(2*z_v[k])))+ \
                        (sqrt(5)-(5*muB[i,k])/(3*length[k]))*sqrt(0.5*z_v[k]/pi)/length[k]*np.exp(-0.5*muB[i,k]**2/z_v[k]))
                else:
                    Ii *= (1+sqrt(5)*np.abs(zX[i,k])/length[k]+5*zX[i,k]**2/(3*length[k]**2))*np.exp(-sqrt(5)*np.abs(zX[i,k])/length[k])  
            I[i] = Ii
            for j in range( i + 1 ):
                if i==j:
                    Jii = 1.
                    for k in range(d):
                        if z_v[k]!=0:
                            Jii *= Jd0(X[i,k],z_m[k],z_v[k],length[k])
                        else:
                            Iki = (1+sqrt(5)*np.abs(zX[i,k])/length[k]+5*zX[i,k]**2/(3*length[k]**2))*np.exp(-sqrt(5)*np.abs(zX[i,k])/length[k])
                            Jii *= Iki**2
                    J[i,j] = Jii
                else:
                    Jij = 1.
                    for k in range(d):
                        if z_v[k]!=0:
                            Jij *= Jd(X[j,k],X[i,k],z_m[k],z_v[k],length[k])
                        else:
                            Iki = (1+sqrt(5)*np.abs(zX[i,k])/length[k]+5*zX[i,k]**2/(3*length[k]**2))*np.exp(-sqrt(5)*np.abs(zX[i,k])/length[k])
                            Ikj = (1+sqrt(5)*np.abs(zX[j,k])/length[k]+5*zX[j,k]**2/(3*length[k]**2))*np.exp(-sqrt(5)*np.abs(zX[j,k])/length[k])
                            Jij *= (Iki*Ikj)
                    J[i,j] = Jij
                    J[j,i] = J[i,j]
    return I,J

@vectorize([float64(float64)],nopython=True,cache=True,fastmath=True)
def pnorm(x):
    """Compute standard normal CDF.
    """
    return 0.5*(1+erf(x/sqrt(2))) 

@njit(cache=True,fastmath=True)
def Jd(X1,X2,z_m,z_v,length):
    """Compute J components in different input dimensions for Matern2.5 kernel.
    """
    if X1<X2:
        x1=X1
        x2=X2
    else:
        x1=X2
        x2=X1
    E30=1+(25*x1**2*x2**2-3*sqrt(5)*(3*length**3+5*length*x1*x2)*(x1+x2)+15*length**2*(x1**2+x2**2+3*x1*x2))/(9*length**4)
    E31=(18*sqrt(5)*length**3+15*sqrt(5)*length*(x1**2+x2**2)-(75*length**2+50*x1*x2)*(x1+x2)+60*sqrt(5)*length*x1*x2)/(9*length**4)
    E32=5*(5*x1**2+5*x2**2+15*length**2-9*sqrt(5)*length*(x1+x2)+20*x1*x2)/(9*length**4)
    E33=10*(3*sqrt(5)*length-5*x1-5*x2)/(9*length**4)
    E34=25/(9*length**4)
    muC=z_m-2*sqrt(5)*z_v/length
    E3A31=E30+muC*E31+(muC**2+z_v)*E32+(muC**3+3*z_v*muC)*E33+(muC**4+6*z_v*muC**2+3*z_v**2)*E34
    E3A32=E31+(muC+x2)*E32+(muC**2+2*z_v+x2**2+muC*x2)*E33+(muC**3+x2**3+x2*muC**2+muC*x2**2+3*z_v*x2+5*z_v*muC)*E34
    P1=exp((10*z_v+sqrt(5)*length*(x1+x2-2*z_m))/length**2)*(0.5*E3A31*(1+erf((muC-x2)/sqrt(2*z_v)))+\
        E3A32*sqrt(0.5*z_v/pi)*exp(-0.5*(x2-muC)**2/z_v))
    
    E40=1+(25*x1**2*x2**2+3*sqrt(5)*(3*length**3-5*length*x1*x2)*(x2-x1)+15*length**2*(x1**2+x2**2-3*x1*x2))/(9*length**4)
    E41=5*(3*sqrt(5)*length*(x2**2-x1**2)+3*length**2*(x1+x2)-10*x1*x2*(x1+x2))/(9*length**4)
    E42=5*(5*x1**2+5*x2**2-3*length**2-3*sqrt(5)*length*(x2-x1)+20*x1*x2)/(9*length**4)
    E43=-50*(X1+X2)/(9*length**4)
    E44=25/(9*length**4)
    E4A41=E40+z_m*E41+(z_m**2+z_v)*E42+(z_m**3+3*z_v*z_m)*E43+(z_m**4+6*z_v*z_m**2+3*z_v**2)*E44
    E4A42=E41+(z_m+x1)*E42+(z_m**2+2*z_v+x1**2+z_m*x1)*E43+(z_m**3+x1**3+x1*z_m**2+z_m*x1**2+3*z_v*x1+5*z_v*z_m)*E44
    E4A43=E41+(z_m+x2)*E42+(z_m**2+2*z_v+x2**2+z_m*x2)*E43+(z_m**3+x2**3+x2*z_m**2+z_m*x2**2+3*z_v*x2+5*z_v*z_m)*E44
    P2=exp(-sqrt(5)*(x2-x1)/length)*(0.5*E4A41*(erf((x2-z_m)/sqrt(2*z_v))-erf((x1-z_m)/sqrt(2*z_v)))+\
        E4A42*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-z_m)**2/z_v)-E4A43*sqrt(0.5*z_v/pi)*exp(-0.5*(x2-z_m)**2/z_v))

    E50=1+(25*x1**2*x2**2+3*sqrt(5)*(3*length**3+5*length*x1*x2)*(x1+x2)+15*length**2*(x1**2+x2**2+3*x1*x2))/(9*length**4)
    E51=(18*sqrt(5)*length**3+15*sqrt(5)*length*(x1**2+x2**2)+(75*length**2+50*x1*x2)*(x1+x2)+60*sqrt(5)*length*x1*x2)/(9*length**4)
    E52=5*(5*x1**2+5*x2**2+15*length**2+9*sqrt(5)*length*(x1+x2)+20*x1*x2)/(9*length**4)
    E53=10*(3*sqrt(5)*length+5*x1+5*x2)/(9*length**4)
    E54=25/(9*length**4)
    muD=z_m+2*sqrt(5)*z_v/length
    E5A51=E50-muD*E51+(muD**2+z_v)*E52-(muD**3+3*z_v*muD)*E53+(muD**4+6*z_v*muD**2+3*z_v**2)*E54
    E5A52=E51-(muD+x1)*E52+(muD**2+2*z_v+x1**2+muD*x1)*E53-(muD**3+x1**3+x1*muD**2+muD*x1**2+3*z_v*x1+5*z_v*muD)*E54
    P3=exp((10*z_v-sqrt(5)*length*(x1+x2-2*z_m))/length**2)*(0.5*E5A51*(1+erf((x1-muD)/sqrt(2*z_v)))+\
        E5A52*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-muD)**2/z_v))

    jd=P1+P2+P3
    return jd

@njit(cache=True,fastmath=True)
def Jd0(x1,z_m,z_v,length):
    """Compute J components in different input dimensions for Matern2.5 kernel.
    """
    E30=1+(25*x1**4-6*sqrt(5)*(3*length**3+5*length*x1**2)*x1+75*length**2*(x1**2))/(9*length**4)
    E31=(18*sqrt(5)*length**3+90*sqrt(5)*length*x1**2-(150*length**2+100*x1**2)*x1)/(9*length**4)
    E32=5*(30*x1**2+15*length**2-18*sqrt(5)*length*x1)/(9*length**4)
    E33=10*(3*sqrt(5)*length-10*x1)/(9*length**4)
    E34=25/(9*length**4)
    muC=z_m-2*sqrt(5)*z_v/length
    E3A31=E30+muC*E31+(muC**2+z_v)*E32+(muC**3+3*z_v*muC)*E33+(muC**4+6*z_v*muC**2+3*z_v**2)*E34
    E3A32=E31+(muC+x1)*E32+(muC**2+2*z_v+x1**2+muC*x1)*E33+(muC**3+x1**3+x1*muC**2+muC*x1**2+3*z_v*x1+5*z_v*muC)*E34
    P1=exp((10*z_v+sqrt(5)*length*(2*x1-2*z_m))/length**2)*(0.5*E3A31*(1+erf((muC-x1)/sqrt(2*z_v)))+\
        E3A32*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-muC)**2/z_v))

    E50=1+(25*x1**4+6*sqrt(5)*(3*length**3+5*length*x1**2)*x1+75*length**2*(x1**2))/(9*length**4)
    E51=(18*sqrt(5)*length**3+90*sqrt(5)*length*x1**2+(150*length**2+100*x1**2)*x1)/(9*length**4)
    E52=5*(30*x1**2+15*length**2+18*sqrt(5)*length*x1)/(9*length**4)
    E53=10*(3*sqrt(5)*length+10*x1)/(9*length**4)
    E54=25/(9*length**4)
    muD=z_m+2*sqrt(5)*z_v/length
    E5A51=E50-muD*E51+(muD**2+z_v)*E52-(muD**3+3*z_v*muD)*E53+(muD**4+6*z_v*muD**2+3*z_v**2)*E54
    E5A52=E51-(muD+x1)*E52+(muD**2+2*z_v+x1**2+muD*x1)*E53-(muD**3+x1**3+x1*muD**2+muD*x1**2+3*z_v*x1+5*z_v*muD)*E54
    P3=exp((10*z_v-sqrt(5)*length*(2*x1-2*z_m))/length**2)*(0.5*E5A51*(1+erf((x1-muD)/sqrt(2*z_v)))+\
        E5A52*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-muD)**2/z_v))

    jd=P1+P3
    return jd

@njit(cache=True,fastmath=True)
def quad(A,B):
    n = len(A)
    a = 0
    for k in range(n):
        for l in range(k+1):
            if k==l:
                a += A[k,l]*B[k]**2
            else:
                a += 2*A[k,l]*B[l]*B[k]
    return a