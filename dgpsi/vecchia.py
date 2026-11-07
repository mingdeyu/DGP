from numba import njit, prange, config, set_num_threads
import numpy as np
from numpy.random import randn
from math import sqrt, exp, fabs, log
from .functions import _Jij_sexp_scaled, _sexp_kvec_invlen, _compute_denoms_from_vs_scaled, _Jii_matern25, _Jij_matern25, _matern25_I1d, _matern25_kvec_invlen, K_matrix_nb, K_sexp_nb, K_matern25_nb, _m5_log_pair, SQRT5, FIVE_THIRDS
from .functions import _m52opt_workspace, _m52opt_query, _m52opt_query_withz
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
core_num = min(core_num or max_threads, max_threads)
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
def _loo_exclude_inplace(NNarray):
    """Exclude training index i from row i, preserving candidate order.

    NNarray must be a private, writable search result in training-row order,
    with distinct valid indices per row and at least one candidate column.
    It contains one extra candidate. Return a view with one fewer column;
    entries may be shifted in place. Do not pass shared training NN state.
    """
    n, width = NNarray.shape
    for i in range(n):
        if NNarray[i, 0] != i:
            break
    else:
        return NNarray[:, 1:]

    for i in range(n):
        for j in range(width - 1):
            if NNarray[i, j] == i:
                for k in range(j, width - 1):
                    NNarray[i, k] = NNarray[i, k + 1]
                break
    return NNarray[:, :-1]

@njit(cache=True)
def _loo_target_first_inplace(NNarray):
    """Put observation i first in row i; modify and return the same array.

    Expects a fresh writable integer neighbour array for the full training
    batch, with distinct valid IDs per row and candidates ordered by distance.
    If i is present, move it first; otherwise insert i and drop the last
    candidate. Preserve the relative order of all retained non-target IDs.
    Empty batches are accepted; nonempty batches require at least one column.
    Uses constant auxiliary storage and leaves already-correct rows untouched.
    """
    n, width = NNarray.shape
    if n == 0:
        return NNarray
    if width == 0:
        raise ValueError("A nonempty neighbour array needs at least one column")

    for i in range(n):
        carry = NNarray[i, 0]
        if carry == i:
            continue

        NNarray[i, 0] = i
        for j in range(1, width):
            current = NNarray[i, j]
            NNarray[i, j] = carry
            if current == i:
                break
            carry = current

    return NNarray

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
        NNarray = np.fliplr(np.sort(NNarray, axis=1))
        NNarray[:, 0] = np.arange(n, dtype=NNarray.dtype)
        return NNarray

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

    NNarray = np.fliplr(np.sort(NNarray, axis=1))
    NNarray[:, 0] = np.arange(n, dtype=NNarray.dtype)
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
    x = randn(n)
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
PAIR_MAP_MAX_BYTES = 256 * 1024**2


@njit(cache=True)
def _vecchia_pair_codes(nn, count):
    n = len(count)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        k = np.int64(count[i])
        offsets[i + 1] = offsets[i] + k*(k-1)//2
    codes = np.empty(offsets[-1], dtype=np.int64)
    for i in range(n):
        p = offsets[i]
        for a in range(count[i]):
            for b in range(a):
                ia, ib = np.int64(nn[i, a]), np.int64(nn[i, b])
                codes[p] = min(ia, ib)*n + max(ia, ib)
                p += 1
    return codes, offsets


@njit(cache=True)
def _vecchia_pair_index(codes, order, n):
    unique_count = 0
    previous = -1
    for p in order:
        code = codes[p]
        if code != previous:
            unique_count += 1
            previous = code
    first = np.empty(unique_count, dtype=np.int32)
    second = np.empty(unique_count, dtype=np.int32)
    ids = np.empty(len(codes), dtype=np.int32)
    previous = -1
    uid = -1
    for p in order:
        code = codes[p]
        if code != previous:
            uid += 1
            first[uid], second[uid] = code//n, code % n
            previous = code
        ids[p] = uid
    return first, second, ids


def prepare_vecchia_pairs(nn, count):
    """Graph-only map, or None when construction would exceed its memory budget.

    Sorting codes needs 8 bytes per occurrence for codes, 8 for the permutation,
    4 for its inverse, and at most 8 for unique endpoints. No sorted code copy
    or int64 inverse is retained. Snapshots, offsets and count scratch are
    included in the cap. Existing maps must be released before calling this.
    """
    n = len(count)
    fixed_bytes = nn.nbytes + count.nbytes + 8*(n+1) + 24*n
    if fixed_bytes > PAIR_MAP_MAX_BYTES or n > np.iinfo(np.int32).max:
        return None
    sizes = count.astype(np.int64)
    occurrences = int(np.sum(sizes*(sizes-1)//2))
    del sizes
    if 28*occurrences + fixed_bytes > PAIR_MAP_MAX_BYTES:
        return None
    codes, offsets = _vecchia_pair_codes(nn, count)
    order = np.argsort(codes)
    return (*_vecchia_pair_index(codes, order, n), offsets)


@njit(cache=True, parallel=True, fastmath=False)
def vecchia_pair_values(x, first, second, length, name):
    """Fresh correlations with v7's separable Matérn arithmetic and guard."""
    iso = length.size == 1
    inv_length = 1.0/length
    values = np.empty(len(first), dtype=np.float64)
    for p in prange(len(first)):
        ia, ib = first[p], second[p]
        if name == 'sexp':
            distance = 0.0
            for j in range(x.shape[1]):
                delta = (x[ia, j]-x[ib, j])*inv_length[0 if iso else j]
                distance += delta*delta
            values[p] = exp(-distance)
        else:
            product, distance = 1.0, 0.0
            for j in range(x.shape[1]):
                delta = fabs((x[ia, j]-x[ib, j])*inv_length[0 if iso else j])
                product *= (1.0 + SQRT5*delta) + FIVE_THIRDS*delta*delta
                distance += delta
            if distance <= 300.0 and product <= 1e300:
                values[p] = product*exp(-SQRT5*distance)
            else:
                values[p] = _m5_log_pair(x[ia], x[ib], length, 2)
    return values


@njit(cache=True, parallel=True, fastmath=True)
def vecchia_llik_pairs(x, y, nn, count, scale, length, nugget, weights, name,
                       first, second, ids, offsets):
    values = vecchia_pair_values(x, first, second, length, name)
    quad, logdet = 0.0, 0.0
    for i in prange(len(count)):
        k = count[i]
        K = np.empty((k, k), dtype=np.float64)
        yi = np.empty(k, dtype=np.float64)
        p = offsets[i]
        for a in range(k):
            ia = nn[i, a]
            yi[a] = y[ia]
            K[a, a] = 1.0 + nugget*weights[ia]
            for b in range(a):
                value = values[ids[p]]
                K[a, b], K[b, a] = value, value
                p += 1
        L = np.linalg.cholesky(K)
        z = forward_solve_last(L, yi)
        quad += z*z
        logdet += 2.0*log(L[k-1, k-1])
    return -0.5*(logdet + quad/scale)


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
        for task in prange(n // 2):
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
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

        for task in prange(n // 2):
            buf = np.empty(d, dtype=np.float64)
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
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
        for task in prange(n // 2):
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
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

        for task in prange(n // 2):
            buf = np.empty(d, dtype=np.float64)
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
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

@njit(cache=True, inline='never')
def chol_retry_subblock(H, scale):
    """Try unperturbed first; cold fallback uses kernel-scale-relative jitter.

Three trial diagonal additions: 1e-14, 1e-12, 1e-10 times GP scale.
Only H is perturbed, and own observation noise remains separate. Successful
Cholesky is not a guarantee of accurate coefficients when H is ill-conditioned.
"""
    try:
        return np.linalg.cholesky(H)
    except Exception:
        pass
    previous = 0.0
    for relative in (1e-14, 1e-12, 1e-10):
        jitter = relative * scale
        for j in range(H.shape[0]):
            H[j, j] += jitter - previous
        previous = jitter
        try:
            return np.linalg.cholesky(H)
        except Exception:
            pass
    raise np.linalg.LinAlgError('Local covariance failed bounded jitter retries')

@njit(cache=True, parallel=True, fastmath=False)
def U_matrix_ptr(X, NNarray, n_obs, length, nugget, scale, gamma, name):
    if nugget != 0.0:
        raise ValueError('U requires nugget=0')
    n, m = NNarray.shape

    data = np.empty(n * m, dtype=np.float64)

    for i in prange(n):
        idx_mod = np.empty(m, dtype=np.int32)
        diag_add = np.empty(m, dtype=np.float64)

        t = 0
        for j in range(m - 1, -1, -1):
            idx = NNarray[i, j]
            cond = idx >= n_obs
            ia = idx - n_obs if cond else idx

            idx_mod[t] = ia
            diag_add[t] = (0.0 if cond else gamma[ia])
            t += 1

        xi = X[idx_mod, :]

        Ki = scale * K_matrix_nb(xi, length, nugget, name)
        add_to_diag_square(Ki, diag_add)
        v = _U_row(Ki)


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
    for i in range(A.shape[0]):
        A[i, i] += d[i]




_RESOLUTION = np.sqrt(32.0 * np.finfo(np.float64).eps)
_COARSE_RCOND = np.sqrt(np.finfo(np.float64).eps)

def _U_validate_parameters(X, length, scale, gamma):
    if (not np.isfinite(X).all() or not np.isfinite(length).all() or
        np.any(np.asarray(length) <= 0) or not np.isfinite(scale) or scale <= 0 or
        not np.isfinite(gamma).all() or np.any(np.asarray(gamma) <= 0)):
        raise np.linalg.LinAlgError('U factor/solve failed: inputs, scale, length and observation variances must be finite and positive')
    if np.asarray(length).size not in (1, X.shape[1]):
        raise ValueError('U length must be scalar or have one entry per input dimension')

@njit(cache=True, fastmath=False)
def _U_physical_pairs(geometry, NNarray, rows, tolerance):
    """Conservative row-local broad phase; exact norms remain at the boundary.

    A physically near local pair necessarily has globally near endpoints.
    Scanning only local latent pairs avoids a global spatial query and exits
    ordinary pairs at their first separated coordinate. No all-pairs array
    is allocated. An O(n*local_width) mask marks possible pair endpoints;
    exact norms and kernel eligibility are evaluated one row at a time.
    """
    n, d = geometry.shape
    width = NNarray.shape[1]
    possible_endpoints = np.zeros(NNarray.shape, dtype=np.bool_)
    for row in rows:
        for pos_a in range(width):
            a = NNarray[row, width - 1 - pos_a] - n
            if a < 0:
                continue
            for pos_b in range(pos_a):
                b = NNarray[row, width - 1 - pos_b] - n
                if b < 0:
                    continue
                possible = True
                for k in range(d):
                    if abs(geometry[a,k] - geometry[b,k]) > tolerance:
                        possible = False
                        break
                if possible:
                    possible_endpoints[row, pos_a] = True
                    possible_endpoints[row, pos_b] = True
    return possible_endpoints

def _U_near_rows(X, NNarray, length):
    """Identify native rows containing physically close distinct latent inputs.

    Physical tolerance scales with design extent, so changing length scales
    cannot make an otherwise ordinary fitted trajectory eligible. A second
    threshold requires a small kernel distance. All original dimensions enter
    both tests. Observation copies alone do not establish a near pair.
    """
    n = len(X)
    rows = np.flatnonzero(np.sum(NNarray >= n, axis=1) >= 2)
    if not len(rows):
        return {}
    
    
    
    amplitude = float(np.max(np.abs(X)))
    geometry = X / amplitude if amplitude > 0.0 else X
    extent = float(np.linalg.norm(np.ptp(geometry, axis=0)))
    tolerance = _RESOLUTION * extent
    
    
    
    broad_tolerance = max(np.nextafter(tolerance,np.inf),
                          np.sqrt(np.finfo(np.float64).tiny))
    candidates = _U_physical_pairs(geometry,NNarray,rows,broad_tolerance)
    result = {}
    for row in np.flatnonzero(candidates.sum(axis=1) >= 2):
        ids = NNarray[row, ::-1]
        positions = np.flatnonzero(candidates[row])
        found = []
        triggered = False
        for j,pos_a in enumerate(positions):
            a = ids[pos_a] - n
            for pos_b in positions[:j]:
                b = ids[pos_b] - n
                if np.linalg.norm(geometry[a]-geometry[b]) > tolerance:
                    continue
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    distance = np.linalg.norm((X[a]-X[b])/length)
                
                
                
                if distance <= np.sqrt(_COARSE_RCOND):
                    found.append((int(pos_a),int(pos_b)))
                if distance <= _RESOLUTION:
                    triggered = True
        if found and triggered:
            result[int(row)] = found
    return result

def _U_coarse_resolved(K, near_pairs, latent):
    
    
    
    
    
    parent = list(range(len(K)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a,b in near_pairs:
        parent[find(b)] = find(a)
    keep = [i for i in range(len(K)) if latent[i] and find(i) == i]
    eig = np.linalg.eigvalsh(K[np.ix_(keep,keep)])
    return bool(np.isfinite(eig).all() and eig[-1] > 0.0 and eig[0] > _COARSE_RCOND * eig[-1])

@njit(cache=True, fastmath=False, inline='never')
def _U_bounded_row(K, scale, max_jitter):
    """Full joint covariance: unperturbed first, then bounded scale-relative jitter.

    Return (coefficients, diagonal addition). The caller also caps the addition
    relative to observation noise. Neither K nor any conditioning variable is
    modified in place. A checked factorization of a perturbed matrix does not
    certify agreement with the original unperturbed statistical model.
    """
    m = len(K)
    if (not np.isfinite(K).all() or not np.isfinite(scale) or scale <= 0.0 or
        not np.isfinite(max_jitter) or max_jitter < 0.0):
        raise np.linalg.LinAlgError('U factor/solve failed: invalid bounded covariance')
    bound = min(max_jitter, 1e-10 * scale)
    previous = -1.0
    for attempt in range(4):
        jitter = 0.0
        if attempt == 1:
            jitter = min(1e-14 * scale, bound)
        elif attempt == 2:
            jitter = min(1e-12 * scale, bound)
        elif attempt == 3:
            jitter = bound
        if jitter <= previous:
            continue
        previous = jitter
        H = K.copy()
        for j in range(m):
            H[j, j] += jitter
        try:
            L = np.linalg.cholesky(H)
            resolved = True
            for j in range(m):
                if not np.isfinite(L[j, j]) or L[j, j] * L[j, j] <= 64.0 * np.finfo(np.float64).eps * m * scale:
                    resolved = False
                    break
            if not resolved:
                continue
            v = np.empty(m)
            
            for j in range(m - 1, -1, -1):
                residual = 1.0 if j == m - 1 else 0.0
                for k in range(j + 1, m):
                    residual -= L[k, j] * v[k]
                v[j] = residual / L[j, j]
            if not np.isfinite(v).all() or v[-1] <= 0.0:
                continue
            
            factor_error = np.max(np.abs(L @ L.T - H))
            matrix_norm = np.max(np.sum(np.abs(H), axis=1))
            rhs = L.T @ v
            rhs[-1] -= 1.0
            solve_norm = np.max(np.sum(np.abs(L.T), axis=1)) * np.max(np.abs(v)) + 1.0
            tolerance = 64.0 * np.finfo(np.float64).eps * m
            if factor_error <= tolerance * matrix_norm and np.max(np.abs(rhs)) <= tolerance * solve_norm:
                return v, jitter
        except Exception:
            pass
    raise np.linalg.LinAlgError('U factor/solve failed: bounded jitter exhausted')

def U_matrix_sp_nb(X, NNarray, scale, length, nugget, name, gamma, rows, cols):
    n = X.shape[0]

    _U_validate_parameters(X, length, scale, gamma)
    data = U_matrix_ptr(X, NNarray, n, length, nugget, scale, gamma, name)
    m = NNarray.shape[1]
    for row, pairs in _U_near_rows(X, NNarray, length).items():
        ids = NNarray[row, ::-1]
        xi = X[ids % n]
        latent = ids >= n
        
        positions = np.flatnonzero(latent)
        for j, a in enumerate(positions):
            for b in positions[:j]:
                if np.array_equal(xi[a], xi[b]):
                    raise np.linalg.LinAlgError('U factor/solve failed: identical latent positions')
        coarse = K_matrix_nb(xi, length, 0.0, name)
        if not _U_coarse_resolved(coarse, pairs, latent):
            continue
        observations = np.where(latent, 0.0, gamma[ids % n])
        covariance = scale * coarse
        add_to_diag_square(covariance, observations)
        bound = 1e-10 * scale
        if np.any(~latent):
            bound = min(bound, 1e-3 * float(np.min(observations[~latent])))
        coefficients, _ = _U_bounded_row(covariance, scale, bound)
        data[row*m:(row+1)*m] = coefficients
    _check_U(data, m)
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
        
        if nugget == 0.0:
            matched = -1
            for a in range(k):
                ia = idx[a]
                same = True
                for j in range(d):
                    if x[i, j] != w[ia, j]:
                        same = False
                        break
                if same:
                    matched = ia
                    break
            if matched >= 0:
                m_out[i] = y[matched, 0]
                v_out[i] = 0.0
                continue

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
        
        if nugget == 0.0:
            matched = -1
            for a in range(k):
                ia = idx[a]
                same = True
                for j in range(d):
                    if x[i, j] != w[ia, j]:
                        same = False
                        break
                if same:
                    matched = ia
                    break
            if matched >= 0:
                m_out[i] = y[matched, 0]
                v_out[i] = 0.0
                continue

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

@njit(cache=True, fastmath=False, inline='never')
def _gp_vecch_deterministic_row(m, z, w1, global_w1, NNarray, y,
                              scale, length, nugget, nugget_diag, name, t):
    """Ordinary GP conditional for one exactly deterministic linked-input row.

    Only selected neighbours are used. Optional z contains deterministic global
    inputs. Strict arithmetic keeps variance independent of the responses.
    """
    k = NNarray.shape[1]
    Dw = w1.shape[1]
    Dz = 0 if z is None else z.shape[1]
    if k == 0:
        return 0.0, scale * (1.0 + nugget)

    if nugget == 0.0:
        for a in range(k):
            ia = NNarray[t, a]
            same = True
            for j in range(Dw):
                if m[t, j] != w1[ia, j]:
                    same = False
                    break
            if same and z is not None:
                for j in range(Dz):
                    if z[t, j] != global_w1[ia, j]:
                        same = False
                        break
            if same:
                return y[ia, 0], 0.0

    Xi = np.empty((k + 1, Dw + Dz), dtype=np.float64)
    yi = np.empty(k, dtype=np.float64)
    for a in range(k):
        ia = NNarray[t, a]
        yi[a] = y[ia, 0]
        for j in range(Dw):
            Xi[a, j] = w1[ia, j]
        if z is not None:
            for j in range(Dz):
                Xi[a, Dw + j] = global_w1[ia, j]
    for j in range(Dw):
        Xi[k, j] = m[t, j]
    if z is not None:
        for j in range(Dz):
            Xi[k, Dw + j] = z[t, j]

    Ki = K_matrix_nb(Xi, length, 0.0, name)
    for a in range(k):
        Ki[a, a] += nugget * nugget_diag[NNarray[t, a]]
    
    
    L = np.linalg.cholesky(Ki[:k, :k])
    c = Ki[k, :k].copy()
    forward_solve_inplace(L, yi, k)
    forward_solve_inplace(L, c, k)
    mean = 0.0
    explained = 0.0
    for a in range(k):
        mean += c[a] * yi[a]
        explained += c[a] * c[a]
    variance = 1.0 + nugget - explained
    if variance < 0.0:
        
        tolerance = 16.0 * np.finfo(np.float64).eps * k * (1.0 + nugget + explained)
        if variance < -tolerance:
            raise ValueError("Negative deterministic GP conditional variance")
        variance = 0.0
    return mean, scale * variance

@njit(cache=False, fastmath=True)
def link_gp_vecch_sexp_noz_serial(m, v, w1, NNarray, y,
                                 scale, length_w, inv_len_w,
                                 nugget, nugget_diag):
    n_pred, d = m.shape
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    k = NNarray.shape[1]

    for t in range(n_pred):
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, None, w1, None, NNarray, y, scale, length_w,
                nugget, nugget_diag, "sexp", t)
            continue

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
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, None, w1, None, NNarray, y, scale, length_w,
                nugget, nugget_diag, "sexp", t)
            continue

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
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, z, w1, global_w1, NNarray, y, scale, length_full,
                nugget, nugget_diag, "sexp", t)
            continue

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
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, z, w1, global_w1, NNarray, y, scale, length_full,
                nugget, nugget_diag, "sexp", t)
            continue

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

@njit(cache=True, fastmath=False)
def _m52vecch_guarded_zero(wi, mean, variance, length):
    """Keep the scalar log-kernel guard when a deterministic t*t may overflow.

    Positive variances, including arbitrarily small positive values, always
    retain the prepared moment route. No input distribution is rounded.
    """
    for d in range(len(mean)):
        if variance[d] == 0.0:
            for i in range(wi.shape[0]):
                if abs((mean[d] - wi[i, d]) / length[d]) > 1e150:
                    return True
    return False


@njit(cache=True, fastmath=False)
def _m52vecch_scalar_contract(mean, variance, wi, inverse, alpha, length, weights):
    """Preserved scalar contraction for extreme deterministic coordinates only."""
    inv_len = 1.0 / length
    inv_len2 = inv_len * inv_len
    len2 = length * length
    len3 = len2 * length
    len4 = len2 * len2
    IR = 0.0
    quad = 0.0
    tr = 0.0
    for i in range(wi.shape[0]):
        Ii = 1.0
        for d in range(len(mean)):
            Ii *= _matern25_I1d(wi[i, d], mean[d], variance[d],
                               length[d], inv_len[d], inv_len2[d])
        si = weights[i]
        ai = alpha[i]
        IR += (Ii * si) * ai
        Jii = _Jii_matern25(wi, i, mean, variance, length,
                           inv_len, inv_len2, len2, len3, len4) * (si * si)
        tr += inverse[i, i] * Jii
        quad += Jii * ai * ai
        for j in range(i):
            Jij = _Jij_matern25(wi, i, j, mean, variance, length,
                               inv_len, inv_len2, len2, len3, len4) * (si * weights[j])
            tr += 2.0 * inverse[i, j] * Jij
            quad += 2.0 * Jij * ai * alpha[j]
    return IR, quad, tr


@njit(cache=False, fastmath=True)
def link_gp_vecch_matern25_noz_serial(m, v, w1, NNarray, y, scale, length_w, nugget, nugget_diag):

    n_pred, d = m.shape
    Dw = w1.shape[1]
    k = NNarray.shape[1]

    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)


    
    workspace = _m52opt_workspace(k, d)
    for t in range(n_pred):
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, None, w1, None, NNarray, y, scale, length_w,
                nugget, nugget_diag, "matern2.5", t)
            continue

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

        
        if _m52vecch_guarded_zero(wi, m[t], v[t], length_w):
            IR, quad, tr = _m52vecch_scalar_contract(
                m[t], v[t], wi, Kinv, alpha, length_w, np.ones(k))
        else:
            IR, quad, tr = _m52opt_query(
                m[t], v[t], wi, Kinv, alpha, length_w, workspace)

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


    for t in prange(n_pred):
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, None, w1, None, NNarray, y, scale, length_w,
                nugget, nugget_diag, "matern2.5", t)
            continue

        
        workspace = _m52opt_workspace(k, d)
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

        
        if _m52vecch_guarded_zero(wi, m[t], v[t], length_w):
            IR, quad, tr = _m52vecch_scalar_contract(
                m[t], v[t], wi, Kinv, alpha, length_w, np.ones(k))
        else:
            IR, quad, tr = _m52opt_query(
                m[t], v[t], wi, Kinv, alpha, length_w, workspace)

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


    
    workspace = _m52opt_workspace(k, d)
    for t in range(n_pred):
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, z, w1, global_w1, NNarray, y, scale, length_full,
                nugget, nugget_diag, "matern2.5", t)
            continue

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

        
        if _m52vecch_guarded_zero(wi, m[t], v[t], length_w):
            weights = _matern25_kvec_invlen(gi, z[t], inv_len_z)
            IR, quad, tr = _m52vecch_scalar_contract(
                m[t], v[t], wi, Kinv, alpha, length_w, weights)
        else:
            IR, quad, tr = _m52opt_query_withz(
                m[t], v[t], z[t], wi, gi, Kinv, alpha,
                length_w, inv_len_z, workspace)

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


    for t in prange(n_pred):
        
        for kk in range(d):
            if v[t, kk] != 0.0:
                break
        else:
            m_new[t], v_new[t] = _gp_vecch_deterministic_row(
                m, z, w1, global_w1, NNarray, y, scale, length_full,
                nugget, nugget_diag, "matern2.5", t)
            continue

        
        workspace = _m52opt_workspace(k, d)
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

        
        if _m52vecch_guarded_zero(wi, m[t], v[t], length_w):
            weights = _matern25_kvec_invlen(gi, z[t], inv_len_z)
            IR, quad, tr = _m52vecch_scalar_contract(
                m[t], v[t], wi, Kinv, alpha, length_w, weights)
        else:
            IR, quad, tr = _m52opt_query_withz(
                m[t], v[t], z[t], wi, gi, Kinv, alpha,
                length_w, inv_len_z, workspace)

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

@njit(cache=True, fastmath=False, inline='never')
def _U_row(K):
    m = len(K)
    v = np.empty(m)
    try:
        L = np.linalg.cholesky(K)
        for j in range(m):
            if not np.isfinite(L[j, j]) or L[j, j] <= 0.0:
                return np.full(m, np.nan)
        solve_lt_e_last_inplace(L, v, m)
        if not np.isfinite(v).all():
            return np.full(m, np.nan)
        return v
    except Exception:
        return np.full(m, np.nan)

def _check_U(data, m):
    bad = np.flatnonzero(~np.isfinite(data))
    if bad.size:
        raise np.linalg.LinAlgError('U factor/solve failed: row ' + str(int(bad[0] // m)))
