from math import erfc as _m52opt_erfc
from numba import njit, prange, config, set_num_threads
import numpy as np
from math import erf, sqrt, pi, exp, fabs
from numpy.random import randn
from scipy.linalg import pinvh, cholesky, solve_triangular, LinAlgError
from scipy.special import betaln, gammaln
import itertools
from psutil import cpu_count
from functools import lru_cache

core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num or max_threads, max_threads)
config.THREADING_LAYER = 'tbb'
set_num_threads(core_num)

SQRT5 = 2.2360679774997898
FIVE_THIRDS = 1.6666666666666667  # 5/3
INV_SQRT2 = 0.7071067811865475
INV_SQRT2PI = 0.3989422804014327  # 1/sqrt(2*pi)

@njit(cache=True)
def g(coef1, coef2, x, name):
    if name=='ga':
        return np.sum(coef1*np.log(x)-coef2*x)
    else:
        return np.sum(-coef1*np.log(x)-coef2/x)

######functions for imputer########
def fmvn(cov, scale):
    """Generate multivariate Gaussian random samples without means.
    """
    L = cholesky(cov, lower=True, check_finite=False)
    return mvn_from_chol_nb(L, sqrt(float(scale[0])))

@njit(cache=True)
def mvn_from_chol_nb(L, sqrt_scale):
    """Generate multivariate Gaussian random samples without means.
    """
    n = L.shape[0]
    sn = randn(n)
    out = L @ sn
    return sqrt_scale * out

@njit(cache=True)
def update_f(f,nu,theta):
    """Update ESS proposal samples.
    """
    fp=f*np.cos(theta) + nu*np.sin(theta)
    return fp

@njit(cache=True)
def logdet_nb(L):
    return 2*np.sum(np.log(np.abs(np.diag(L))))

######Gauss-Hermite quadrature######
# def ghdiag(fct,mu,var,y):
#     x, w = np.polynomial.hermite.hermgauss(10)
#     N = np.shape(mu)[1]
#     const = np.pi**(-0.5*N)
#     xn = np.array(list(itertools.product(*(x,)*N)))
#     wn = np.prod(np.array(list(itertools.product(*(w,)*N))), 1)[:, None]
#     fn = sqrt(2.0)*(np.sqrt(var[:,None])*xn) + mu[:,None]
#     llik=fct(y[:,None],fn)
#     return np.sum(np.exp(np.log((wn * const)[None,:]) + llik), axis=1)


_GH_WORKSPACE_BYTES = 8 * 1024 * 1024
_GH_GRID_ENTRY_BYTES = 512 * 1024


@lru_cache(maxsize=8)
def _hermgauss_1d(q: int):
    x, w = np.polynomial.hermite.hermgauss(q)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def _gh_tensor_block(N, q, start, stop):
    """Enumerate a bounded tensor block in itertools.product order."""
    x, w = _hermgauss_1d(q)
    logw1 = np.log(w) - 0.5 * np.log(np.pi)
    index = np.arange(start, stop, dtype=np.int64)
    nodes = np.empty((stop - start, N), dtype=np.float64)
    logw = np.zeros(stop - start, dtype=np.float64)
    for k in range(N - 1, -1, -1):
        digit = index % q
        index //= q
        nodes[:, k] = x[digit]
        logw += logw1[digit]
    return nodes, logw


@lru_cache(maxsize=4)
def _gh_tensor_grid_cached(N, q):
    
    return _gh_tensor_block(N, q, 0, q ** N)


def _gh_tensor_grid(N: int, q: int):
    """Return a tensor grid; cache only small grids, in original node order."""
    _hermgauss_1d(q)
    if N == 0:
        raise ValueError('need at least one array to stack')
    if 8 * (q ** N) * (N + 1) <= _GH_GRID_ENTRY_BYTES:
        return _gh_tensor_grid_cached(N, q)
    return _gh_tensor_block(N, q, 0, q ** N)



_gh_tensor_grid.cache_clear = _gh_tensor_grid_cached.cache_clear
_gh_tensor_grid.cache_info = _gh_tensor_grid_cached.cache_info


def _gh_node_blocks(N, q, block):
    P = q ** N
    if 8 * P * (N + 1) <= _GH_GRID_ENTRY_BYTES:
        nodes, logw = _gh_tensor_grid_cached(N, q)
        for j in range(0, P, block):
            yield nodes[j:j + block], logw[j:j + block]
    else:
        for j in range(0, P, block):
            yield _gh_tensor_block(N, q, j, min(j + block, P))


def _logsumexp_axis1(a: np.ndarray) -> np.ndarray:
    """Stable logsumexp over axis 1, including impossible contributions."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], -np.inf, dtype=np.float64)
    amax = np.max(a, axis=1, keepdims=True)
    
    amax = np.where(np.isfinite(amax), amax, 0.0)
    with np.errstate(under='ignore', over='ignore', divide='ignore', invalid='ignore'):
        return amax[:, 0] + np.log(np.sum(np.exp(a - amax), axis=1))


def _gh_log_values(fct, y, latent):
    values = np.asarray(fct(y[:, None], latent))
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    if values.ndim != 2:
        raise ValueError(f'Unexpected llik shape {values.shape}; expected (M,B) or (M,B,1).')
    return values


def _gh_integrate_group(fct, mu, var, y, active, q, block):
    """Bound row/node intermediates; callbacks still receive all coordinates."""
    M, N = mu.shape
    D = len(active)
    B = min(block, q ** D)
    width = max(1, min(M, _GH_WORKSPACE_BYTES // (8 * B * (2 * N + 6))))
    total = np.full(M, -np.inf, dtype=np.float64)
    sqrt_var = np.sqrt(var)
    sqrt2 = np.sqrt(2.0)
    deterministic = np.ones(N, dtype=bool)
    deterministic[active] = False
    fixed = np.flatnonzero(deterministic)
    for nodes, logw in _gh_node_blocks(D, q, block):
        for first in range(0, M, width):
            last = min(first + width, M)
            latent = np.empty((last - first, len(nodes), N), dtype=np.float64)
            for k in fixed:
                latent[:, :, k] = mu[first:last, k, None]
            for j, k in enumerate(active):
                np.multiply(sqrt_var[first:last, k, None], nodes[None, :, j], out=latent[:, :, k])
                latent[:, :, k] *= sqrt2
                latent[:, :, k] += mu[first:last, k, None]
            values = _gh_log_values(fct, y[first:last], latent)
            total[first:last] = np.logaddexp(total[first:last], _logsumexp_axis1(values + logw))
    return total


def _ghdiag_log(fct, mu, var, y, q: int = 10, block: int = 8192, max_nodes: int = 2_000_000):
    """Log of diagonal Gaussian GH integral; pointwise likelihood callbacks."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    y = np.asarray(y)
    M, N = mu.shape
    P = q ** N
    
    if P > max_nodes:
        raise ValueError(
            f'Tensor GH grid too large: q**N = {q}^{N} = {P} nodes. '
            'Increase max_nodes, reduce N/q, or switch to a different approximation.'
        )
    _hermgauss_1d(q)  
    if N == 0:
        raise ValueError('need at least one array to stack')
    node_range = range(0, P, block)  
    if not node_range or M == 0:
        return np.full(M, -np.inf, dtype=np.float64)
    if var.ndim < 2:
        var[:, None, :]  
    if var.shape != mu.shape:
        var = np.broadcast_to(var, mu.shape)
    if y.shape[0] == 1 and M != 1:
        y = np.broadcast_to(y, (M,) + y.shape[1:])

    zero = var == 0.0
    if zero.any():
        if zero.all():
            if 8 * M * (2 * N + 6) > _GH_WORKSPACE_BYTES:
                return _gh_integrate_group(fct, mu, var, y, np.empty(0, dtype=int), q, block)
            values = _logsumexp_axis1(_gh_log_values(fct, y, mu[:, None, :].copy()))
            return np.broadcast_to(values, (M,)).copy()
        total = np.full(M, -np.inf, dtype=np.float64)
        masks, groups = np.unique(zero, axis=0, return_inverse=True)
        for i, fixed in enumerate(masks):
            rows = np.flatnonzero(groups == i)
            means, variances, observations = mu[rows], var[rows], y[rows]
            active = np.flatnonzero(~fixed)
            if len(active) == 0 and 8 * len(rows) * (2 * N + 6) <= _GH_WORKSPACE_BYTES:
                total[rows] = _logsumexp_axis1(_gh_log_values(fct, observations, means[:, None, :].copy()))
            else:
                total[rows] = _gh_integrate_group(fct, means, variances, observations, active, q, block)
        return total

    estimate = 8 * M * min(P, block) * (2 * N + 6)
    if estimate > _GH_WORKSPACE_BYTES:
        return _gh_integrate_group(fct, mu, var, y, np.arange(N), q, block)

    
    sqrt_var = np.sqrt(var)
    sqrt2 = np.sqrt(2.0)
    if P <= block and 8 * P * (N + 1) <= _GH_GRID_ENTRY_BYTES:
        nodes, logw = _gh_tensor_grid_cached(N, q)
        latent = sqrt2 * (sqrt_var[:, None, :] * nodes[None, :, :]) + mu[:, None, :]
        values = _logsumexp_axis1(_gh_log_values(fct, y, latent) + logw)
        return values if values.shape == (M,) else np.broadcast_to(values, (M,)).copy()
    total = np.full(M, -np.inf, dtype=np.float64)
    for nodes, logw in _gh_node_blocks(N, q, block):
        latent = sqrt2 * (sqrt_var[:, None, :] * nodes[None, :, :]) + mu[:, None, :]
        log_block = _logsumexp_axis1(_gh_log_values(fct, y, latent) + logw)
        total = np.logaddexp(total, log_block)
    return total


def ghdiag(fct, mu, var, y, q: int = 10, block: int = 8192, max_nodes: int = 2_000_000):
    """GH expectation of exp(loglik(y | f)); retains the probability interface."""
    return np.exp(_ghdiag_log(fct, mu, var, y, q, block, max_nodes))


######MICE smooth pred var calculation######
def mice_var(x, x_extra, input_dim, connect, name, length, scale, nugget, nugget_s):
    """Calculate smoothed predictive variances of the GP using the candidate design set.
    """
    kernel_input=x[:,input_dim]
    if connect is not None:
        kernel_global_input=x_extra[:,connect]
        kernel_input=np.concatenate((kernel_input, kernel_global_input),1)
    kernel_nugget=max(nugget_s,nugget)
    R=K_matrix_nb(kernel_input, length, kernel_nugget, name)
    n = R.shape[0]
    try:
        L = cholesky(R, lower=True, check_finite=False)
        invL = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
        diag_Rinv = np.sum(invL * invL, axis=0)
        sigma2 = (scale / diag_Rinv).reshape(-1, 1)
        return sigma2
    except LinAlgError:
        Rinv=pinvh(R,check_finite=False)
        sigma2 = (scale/np.diag(Rinv)).reshape(-1,1)
        return sigma2

######helper functions for predictions########
@njit(cache=True, fastmath=True)
def cond_mean_core(X, Z, Rinv_y, length, name):
    n_pred, d = Z.shape
    X_l = X / length         
    out = np.empty(n_pred, dtype=np.float64)
    z_l = np.empty(d, dtype=np.float64)
    
    for t in range(n_pred):
        z_l[:] = Z[t] / length  
        kvec = K_vec_nb(X_l, z_l, name)
        out[t] = np.dot(Rinv_y, kvec)

    return out

def cond_mean(x, z, w1, global_w1, Rinv_y, length, name):
    """Make GP predictions."""
    if z is not None:
        x = np.concatenate((x, z), axis=1)
        w1 = np.concatenate((w1, global_w1), axis=1)

    x = np.ascontiguousarray(x, dtype=np.float64)
    w1 = np.ascontiguousarray(w1, dtype=np.float64)

    return cond_mean_core(w1, x, Rinv_y, length, name)

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


@njit(cache=True, fastmath=False, inline='never')
def K_matern25_nb(xi, length, nugget):
    n, d = xi.shape
    iso = length.size == 1
    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = nugget.size == 1
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
            for j in range(i):
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi[i, k] - xi[j, k]) * inv)
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                if coef2 <= 300.0 and coef1 <= 1e+300:
                    kij = coef1 * exp(-SQRT5 * coef2)
                else:
                    kij = _m5_log_pair(xi[i], xi[j], length, 2)
                K[i, j] = kij
                K[j, i] = kij
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]
        for i in range(1, n):
            for j in range(i):
                coef1 = 1.0
                coef2 = 0.0
                for k in range(d):
                    distk = fabs((xi[i, k] - xi[j, k]) * inv_ell[k])
                    el1 = 1.0 + SQRT5 * distk
                    el2 = FIVE_THIRDS * distk * distk
                    coef = el1 + el2
                    coef1 *= coef
                    coef2 += distk
                if coef2 <= 300.0 and coef1 <= 1e+300:
                    kij = coef1 * exp(-SQRT5 * coef2)
                else:
                    kij = _m5_log_pair(xi[i], xi[j], length, 2)
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
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]

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
                        diff = (xi_i[k] - xi_j[k]) * inv_ell[k]
                        t = diff * diff
                        dist += t
                    kij = exp(-dist)
                    K[i, j] = kij
                    K[j, i] = kij
    return K


@njit(cache=True, fastmath=False, inline='never', parallel=True)
def K_matern25_nb_parallel(xi, length, nugget):
    n, d = xi.shape
    iso = length.size == 1
    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = nugget.size == 1
    if nug_scalar:
        ng0 = nugget[0]
        for i in range(n):
            K[i, i] = 1.0 + ng0
    else:
        for i in range(n):
            K[i, i] = 1.0 + nugget[i]
    if iso:
        inv = 1.0 / length[0]
        for task in prange(n // 2):
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
                for j in range(i):
                    coef1 = 1.0
                    coef2 = 0.0
                    for k in range(d):
                        distk = fabs((xi[i, k] - xi[j, k]) * inv)
                        el1 = 1.0 + SQRT5 * distk
                        el2 = FIVE_THIRDS * distk * distk
                        coef = el1 + el2
                        coef1 *= coef
                        coef2 += distk
                    if coef2 <= 300.0 and coef1 <= 1e+300:
                        kij = coef1 * exp(-SQRT5 * coef2)
                    else:
                        kij = _m5_log_pair(xi[i], xi[j], length, 2)
                    K[i, j] = kij
                    K[j, i] = kij
    else:
        inv_ell = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv_ell[k] = 1.0 / length[k]
        for task in prange(n // 2):
            low = np.int64(task) + 1
            high = n - low
            for side in range(1 if low == high else 2):
                i = low if side == 0 else high
                for j in range(i):
                    coef1 = 1.0
                    coef2 = 0.0
                    for k in range(d):
                        distk = fabs((xi[i, k] - xi[j, k]) * inv_ell[k])
                        el1 = 1.0 + SQRT5 * distk
                        el2 = FIVE_THIRDS * distk * distk
                        coef = el1 + el2
                        coef1 *= coef
                        coef2 += distk
                    if coef2 <= 300.0 and coef1 <= 1e+300:
                        kij = coef1 * exp(-SQRT5 * coef2)
                    else:
                        kij = _m5_log_pair(xi[i], xi[j], length, 2)
                    K[i, j] = kij
                    K[j, i] = kij
    return K

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


@njit(cache=True, fastmath={'reassoc', 'contract'}, inline='never')
def K_vec_matern25_nb(X, z):
    n1, d = X.shape
    K_vec = np.empty(n1, dtype=np.float64)
    for i in range(n1):
        coef1 = 1.0
        coef2 = 0.0
        for k in range(d):
            distk = fabs(X[i, k] - z[k])
            coef1 *= 1.0 + SQRT5 * distk + FIVE_THIRDS * distk * distk
            coef2 += distk
        if coef2 <= 300.0 and coef1 <= 1e+300:
            K_vec[i] = coef1 * exp(-SQRT5 * coef2)
        else:
            K_vec[i] = _m5_log_pair(X[i], z, z, 0)
    return K_vec

# gp predictions (non-vecchia)

@njit(cache=True)
def gp_non_parallel(x,w1,Rinv,Rinv_y,scale,nugget,name):
    """Make GP predictions
    """
    n_pred = x.shape[0]
    m, v = np.empty(n_pred), np.empty(n_pred)
    for i in range(n_pred):
        ri=K_vec_nb(w1,x[i],name)
        Rinv_ri=np.dot(Rinv,ri)
        r_Rinv_r=np.dot(ri, Rinv_ri)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = abs(scale*(1.0+nugget-r_Rinv_r))
    return m, v


@njit(cache=True)
def gp(x,w1,Rinv,Rinv_y,scale,nugget,name):
    """Make GP predictions with contiguous rows for the symmetric inverse."""
    if Rinv.flags.f_contiguous and not Rinv.flags.c_contiguous:
        return _gp_layout_core(x,w1,Rinv.T,Rinv_y,scale,nugget,name)
    return _gp_layout_core(x,w1,Rinv,Rinv_y,scale,nugget,name)


@njit(cache=True, parallel=True)
def _gp_layout_core(x,w1,Rinv,Rinv_y,scale,nugget,name):
    """Make GP predictions
    """
    n_pred = x.shape[0]
    n_train = w1.shape[0]
    m, v = np.empty(n_pred), np.empty(n_pred)
    for i in prange(n_pred):
        ri=K_vec_nb(w1,x[i],name)
        r_Rinv_r=quad0(Rinv, ri, n_train)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = abs(scale*(1.0+nugget-r_Rinv_r))
    return m, v

@njit(cache=True, fastmath=True)
def quad0(A, B, n):
    s = 0.0
    for k in range(n):
        bk = B[k]
        s += A[k, k] * bk * bk
        for l in range(k):
            s += 2.0 * A[k, l] * bk * B[l]
    return s

# linked GP predictions (non-vecchia, sexp)




@njit(cache=True)
def _seopt_workspace(n,d):
    return np.empty(n),np.empty(n),np.empty((d,n)),np.empty((min(32,n),n))


@njit(cache=True,fastmath=True)
def _seopt_prepare(w1,m_row,v_row,inv_len,I,B,E):
    """I_i and sqrt(J_ii); J_ij=B_i*B_j*exp(-||E_i-E_j||^2).

    E only contains positive-variance coordinates. All deterministic dimensions
    are absorbed in the endpoint factors. Coordinates are centered before scaling.
    """
    n,d=w1.shape
    if len(m_row)!=d or len(v_row)!=d or len(inv_len)!=d:
        raise ValueError('Mean, variance and inverse length must have d entries')
    for i in range(n):I[i]=0.0;B[i]=0.0
    ic=1.0;bc=1.0;active=0
    for k in range(d):
        inv=inv_len[k]
        if v_row[k]<0.0 or inv<=0.0:
            raise ValueError('Require variance>=0 and inverse length>0')
        q=v_row[k]*inv*inv
        di=1.0/(1.0+2.0*q)
        dj=1.0/(1.0+4.0*q)
        ic*=sqrt(di)
        bc*=sqrt(sqrt(dj))
        for i in range(n):
            r=(w1[i,k]-m_row[k])*inv
            rr=r*r
            I[i]+=rr*di
            B[i]+=rr*dj
        if v_row[k]!=0.0:
            c=sqrt(2.0*q*dj)*inv
            anchor=0.0
            if n:
                lo=w1[0,k];hi=w1[0,k]
                for i in range(1,n):lo=min(lo,w1[i,k]);hi=max(hi,w1[i,k])
                anchor=0.5*lo+0.5*hi
            for i in range(n):E[active,i]=(w1[i,k]-anchor)*c
            active+=1
    for i in range(n):I[i]=ic*exp(-I[i]);B[i]=bc*exp(-B[i])
    return active


@njit(cache=True,fastmath=True)
def _seopt_contract(I,B,E,tile,active,Rinv,Rinv_y):
    n=len(I);IR=0.0;quad=0.0;tr=0.0
    for i in range(n):
        IR+=I[i]*Rinv_y[i]
        diagonal=B[i]*B[i]
        tr+=Rinv[i,i]*diagonal
        quad+=diagonal*Rinv_y[i]*Rinv_y[i]
    if active==0:
        for i in range(n):
            for j in range(i):tr+=2.0*Rinv[i,j]*I[i]*I[j]
        return IR,IR*IR,tr
    if active==1:
        for i in range(n):
            for j in range(i):
                delta=E[0,i]-E[0,j]
                Jij=B[i]*B[j]*exp(-delta*delta)
                tr+=2.0*Rinv[i,j]*Jij
                quad+=2.0*Jij*Rinv_y[i]*Rinv_y[j]
        return IR,quad,tr
    if n==0:return IR,quad,tr
    for ib in range(0,n,tile.shape[0]):
        ie=min(n,ib+tile.shape[0])
        for i in range(ib,ie):
            for j in range(i):tile[i-ib,j]=0.0
        for k in range(active):
            for i in range(ib,ie):
                for j in range(i):
                    delta=E[k,i]-E[k,j]
                    tile[i-ib,j]+=delta*delta
        for i in range(ib,ie):
            for j in range(i):
                Jij=B[i]*B[j]*exp(-tile[i-ib,j])
                tr+=2.0*Rinv[i,j]*Jij
                quad+=2.0*Jij*Rinv_y[i]*Rinv_y[j]
    return IR,quad,tr


@njit(cache=True)
def _seopt_query(m_row,v_row,w1,Rinv,Rinv_y,inv_len_w,workspace):
    I,B,E,tile=workspace
    active=_seopt_prepare(w1,m_row,v_row,inv_len_w,I,B,E)
    return _seopt_contract(I,B,E,tile,active,Rinv,Rinv_y)


@njit(cache=True)
def _seopt_query_withz(m_row,v_row,z_row,w1,global_w1,Rinv,Rinv_y,inv_len_w,inv_len_z,workspace):
    I,B,E,tile=workspace
    active=_seopt_prepare(w1,m_row,v_row,inv_len_w,I,B,E)
    zi=_sexp_kvec_invlen(global_w1,z_row,inv_len_z)
    for i in range(len(I)):I[i]*=zi[i];B[i]*=zi[i]
    return _seopt_contract(I,B,E,tile,active,Rinv,Rinv_y)


@njit(cache=True, fastmath=True, inline="always")
def _sexp_kvec_invlen(X, z, inv_len):
    """Compute exp(-|| (X-z)/length ||^2) using inv_len; returns (n,)"""
    n, d = X.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist = 0.0
        for k in range(d):
            diff = (X[i, k] - z[k]) * inv_len[k]
            dist += diff * diff
        out[i] = exp(-dist)
    return out

@njit(cache=True, fastmath=True, inline="always")
def _compute_denoms_from_vs_scaled(vs, denI, denJ):
    """
    vs is already scaled: vs[k] = z_v[k] / length[k]^2
    div = 2*vs;  denI = 1/(1+div); denJ = 1/(2+4*div) = 1/(2+8*vs)
    Returns (I_coef1, J_coef1).
    """
    d = vs.shape[0]
    I_prod = 1.0
    J_prod = 1.0
    for k in range(d):
        div = 2.0 * vs[k]
        I_prod *= (1.0 + div)
        J_prod *= (1.0 + 2.0 * div)               # == (1 + 4*vs)
        denI[k] = 1.0 / (1.0 + div)
        denJ[k] = 1.0 / (2.0 + 4.0 * div)         # == 1/(2 + 8*vs)
    return 1.0 / sqrt(I_prod), 1.0 / sqrt(J_prod)

@njit(cache=True, fastmath=True, inline="always")
def _Jij_sexp_scaled(w1, i, j, ms_scaled, denJ, inv_len_w, J_coef1):
    """
    Compute Jij for sexp using:
      Jij = J_coef1 * exp( - sum_k [ 0.5*(xi-xj)^2 + (xi+xj-2*ms)^2 * denJ[k] ] )
    where xi = w1[i,k]/len, ms = m/len, denJ = 1/(2+8*vs).
    """
    d = ms_scaled.shape[0]
    expo = 0.0
    for k in range(d):
        xi = w1[i, k] * inv_len_w[k]
        xj = w1[j, k] * inv_len_w[k]

        diff = xi - xj
        expo += 0.5 * diff * diff

        s = (xi + xj) - 2.0 * ms_scaled[k]
        expo += s * s * denJ[k]
    return J_coef1 * exp(-expo)


@njit(cache=True,fastmath=True)
def link_gp_sexp_noz_serial(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget):
    p,d=m.shape
    means=np.empty(p);variances=np.empty(p)
    ws=_seopt_workspace(len(w1),d)
    for t in range(p):
        IR,quad,tr=_seopt_query(m[t],v[t],w1,Rinv,Rinv_y,inv_len_w,ws)
        means[t]=IR;variances[t]=abs(quad-IR*IR+scale*(1.0+nugget-tr))
    return means,variances




@njit(cache=True,fastmath=True,parallel=True)
def link_gp_sexp_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget):
    p,d=m.shape
    means=np.empty(p);variances=np.empty(p)
    for t in prange(p):
        ws=_seopt_workspace(len(w1),d)
        IR,quad,tr=_seopt_query(m[t],v[t],w1,Rinv,Rinv_y,inv_len_w,ws)
        means[t]=IR;variances[t]=abs(quad-IR*IR+scale*(1.0+nugget-tr))
    return means,variances



@njit(cache=True,fastmath=True)
def link_gp_sexp_withz_serial(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget):
    p,d=m.shape
    means=np.empty(p);variances=np.empty(p)
    ws=_seopt_workspace(len(w1),d)
    for t in range(p):
        IR,quad,tr=_seopt_query_withz(m[t],v[t],z[t],w1,global_w1,Rinv,Rinv_y,inv_len_w,inv_len_z,ws)
        means[t]=IR;variances[t]=abs(quad-IR*IR+scale*(1.0+nugget-tr))
    return means,variances



@njit(cache=True,fastmath=True,parallel=True)
def link_gp_sexp_withz_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget):
    p,d=m.shape
    means=np.empty(p);variances=np.empty(p)
    for t in prange(p):
        ws=_seopt_workspace(len(w1),d)
        IR,quad,tr=_seopt_query_withz(m[t],v[t],z[t],w1,global_w1,Rinv,Rinv_y,inv_len_w,inv_len_z,ws)
        means[t]=IR;variances[t]=abs(quad-IR*IR+scale*(1.0+nugget-tr))
    return means,variances


# linked GP predictions (non-vecchia, matern2.5)

@njit(cache=True, fastmath=True, inline="always")
def _Phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + erf(x * INV_SQRT2))



@njit(cache=True, fastmath=False, inline='never')
def _matern25_det_1d(absdiff_scaled):
    r = absdiff_scaled
    if r <= 300.0:
        return (1.0 + SQRT5 * r + FIVE_THIRDS * r * r) * exp(-SQRT5 * r)
    return exp(-_m5_cost(SQRT5 * r))



@njit(cache=True, fastmath={'reassoc', 'contract'}, inline='never')
def _matern25_kvec_invlen(X, z, inv_len):
    """
    Deterministic Matérn-2.5 kvec (used for global z part, SAME role as K_vec_nb),
    but uses inv_len to avoid divides.
    """
    n, d = X.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        coef1 = 1.0
        coef2 = 0.0
        for k in range(d):
            distk = abs((X[i, k] - z[k]) * inv_len[k])
            coef1 *= 1.0 + SQRT5 * distk + FIVE_THIRDS * distk * distk
            coef2 += distk
        if coef2 <= 300.0 and coef1 <= 1e+300:
            out[i] = coef1 * exp(-SQRT5 * coef2)
        else:
            out[i] = _m5_log_pair(X[i], z, inv_len, 1)
    return out



@njit(cache=True, fastmath=False)
def _m52candidate_cf(r,s,ss,density,c):
    """Positive convergents; adjacent truncations bracket the exact R4.

    R4=4/(b+5/(b+6/(...))). Positive continuants avoid subtractive
    forward moment recurrences. The determinant gives the bracket width
    without subtracting convergents. Ratios R1..R3 are then positive.
    Work is capped at the existing maximum depth, with explicit failure.
    """
    b=-(r-c*ss)/s
    lower4=4.0/(b+5.0/b)
    if b>=1.0e9:
        
        
        r4=4.0/b
    else:
        pold=0.0;pcur=4.0;qold=1.0;qcur=b;det=4.0
        converged=False
        for depth in range(5,193):
            pnew=b*pcur+depth*pold
            qnew=b*qcur+depth*qold
            det*=depth
            pold=pcur;pcur=pnew;qold=qcur;qcur=qnew
            if depth%4==0:
                width=det/qcur/qold
                
                
                
                if 20.0*width<=2.0e-15*lower4:
                    converged=True
                    break
                if qcur>1.0e70:
                    pold*=1.0e-70;pcur*=1.0e-70
                    qold*=1.0e-70;qcur*=1.0e-70;det*=1.0e-140
        if not converged:
            raise ValueError('Matérn positive tail fraction did not converge')
        r4=pcur/qcur
    r3=3.0/(b+r4);r2=2.0/(b+r3);r1=1.0/(b+r2)
    f0=density/(b+r1)
    f1=f0*(s*r1);f2=f1*(s*r2);f3=f2*(s*r3);f4=f3*(s*r4)
    if not (np.isfinite(f0) and np.isfinite(f1) and np.isfinite(f2)
            and np.isfinite(f3) and np.isfinite(f4)):
        raise ValueError('Matérn positive tail moments are not finite')
    return f0,f1,f2,f3,f4
















@njit(cache=True,fastmath=False,inline='always')
def _m52opt_tilted(r,s,ss,density,c):
    """Tilted positive-half-line moments, evaluated without saturated 1+erf.

    For wide, left-shifted tails use the positive Laplace continued fraction
    H_n/H_(n-1)=n/(b+H_(n+1)/H_n). This avoids forward cancellation. It evaluates
    the same analytic moments; no input distribution/kernel approximation.
    """
    u=r-c*ss
    if u < -2.0*s and ss>0.25:
        return _m52candidate_cf(r,s,ss,density,c)
    
    
    if u<0.0 and density==0.0:return 0.0,0.0,0.0,0.0,0.0
    f0=exp(0.5*c*c*ss-c*r)*0.5*_m52opt_erfc(-u/(sqrt(2.0)*s))
    f1=u*f0+s*density
    f2=u*f1+ss*f0
    f3=u*f2+2.0*ss*f1
    f4=u*f3+3.0*ss*f2
    return f0,f1,f2,f3,f4


@njit(cache=True,fastmath=True,inline='always')
def _m52opt_tail(r,s,ss,density):
    return _m52opt_tilted(r,s,ss,density,2.0)



@njit(cache=True,inline='always')
def _m52opt_xi(r,s,ss,density):
    p0,p1,p2,p3,p4=_m52opt_tilted(r,s,ss,density,1.0)
    m0,m1,m2,m3,m4=_m52opt_tilted(-r,s,ss,density,1.0)
    return p0+m0+p1+m1+(p2+m2)/3.0



@njit(cache=True, fastmath=False)
def _m52opt_prepare(X, mean, variance, length, I, C, S, D, diag, factor_exponentials):
    n, d = X.shape
    if len(mean) != d or len(variance) != d or len(length) != d:
        raise ValueError('mean, variance and length must have d entries')
    for i in range(n):
        I[i] = 1.0
        D[i] = 1.0
        diag[i] = 1.0
    active = 0
    for k in range(d):
        if variance[k] < 0.0 or length[k] <= 0.0:
            raise ValueError('Require variance>=0 and length>0')
        lam = sqrt(5.0) / length[k]
        if variance[k] == 0.0:
            for i in range(n):
                t = lam * abs(mean[k] - X[i, k])
                value = (1.0 + t + t * t / 3.0) * exp(-t)
                I[i] *= value
                D[i] *= value
            continue
        ss = lam * lam * variance[k]
        s = sqrt(ss)
        if n:
            xmin = X[0, k]
            xmax = X[0, k]
            for i in range(1, n):
                xmin = min(xmin, X[i, k])
                xmax = max(xmax, X[i, k])
            span = lam * (xmax - xmin)
            anchor = 0.5 * xmin + 0.5 * xmax
        else:
            span = 0.0
            anchor = 0.0
        factor = factor_exponentials and np.isfinite(span) and (span <= 600.0)
        S[active, 0] = s
        S[active, 1] = ss
        S[active, 2] = 1.0 if factor else 0.0
        S[active, 3] = 3.0 + ss
        S[active, 4] = 9.0 - 3.0 * ss
        S[active, 5] = 6.0 * ss - 3.0
        S[active, 6] = 9.0 - 3.0 * ss + 3.0 * ss * ss
        S[active, 7] = 4.0 * ss
        for i in range(n):
            geometry = lam * (X[i, k] - anchor)
            r = lam * (mean[k] - X[i, k])
            density = exp(-0.5 * r * r / ss) / sqrt(2.0 * pi)
            fp0, fp1, fp2, fp3, fp4 = _m52opt_tail(r, s, ss, density)
            fm0, fm1, fm2, fm3, fm4 = _m52opt_tail(-r, s, ss, density)
            C[active, 0, i] = geometry
            C[active, 1, i] = r
            C[active, 2, i] = 0.5 * erf(-r / (sqrt(2.0) * s))
            C[active, 3, i] = density
            C[active, 4, i] = fp0
            C[active, 5, i] = fp1
            C[active, 6, i] = fp2
            C[active, 7, i] = fp3
            C[active, 8, i] = fp4
            C[active, 9, i] = fm0
            C[active, 10, i] = fm1
            C[active, 11, i] = fm2
            C[active, 12, i] = fm3
            C[active, 13, i] = fm4
            if factor:
                C[active, 14, i] = exp(geometry)
                C[active, 15, i] = exp(-geometry)
            I[i] *= _m52opt_xi(r, s, ss, density)
            diag[i] *= fp0 + fm0 + 2.0 * (fp1 + fm1) + 5.0 / 3.0 * (fp2 + fm2) + 2.0 / 3.0 * (fp3 + fm3) + (fp4 + fm4) / 9.0
        active += 1
    for i in range(n):
        diag[i] *= D[i] * D[i]
    return active


@njit(cache=True, fastmath=True)
def _m52opt_compress(C, S, active):
    """Q_d=A+d*B+d^2*C; precombine its coefficients once per endpoint.

    Store P=s*phi(r/s) and r*P for the factored middle boundary terms.
    """
    for i in range(C.shape[2]):
        for k in range(active):
            for offset in (4, 9):
                f0 = C[k, offset, i]
                f1 = C[k, offset + 1, i]
                f2 = C[k, offset + 2, i]
                f3 = C[k, offset + 3, i]
                f4 = C[k, offset + 4, i]
                C[k, offset, i] = f0 + 2.0 * f1 + 5.0 / 3.0 * f2 + 2.0 / 3.0 * f3 + f4 / 9.0
                C[k, offset + 1, i] = f0 + 5.0 / 3.0 * f1 + f2 + 2.0 / 9.0 * f3
                C[k, offset + 2, i] = (f0 + f1) / 3.0 + f2 / 9.0
            C[k, 3, i] *= S[k, 0]
            C[k, 7, i] = C[k, 1, i] * C[k, 3, i]


@njit(cache=True, fastmath=True, inline='always')
def _m52opt_pair_compressed(C, S, i, j, k):
    forward = C[k, 0, i] <= C[k, 0, j]
    gap = abs(C[k, 0, j] - C[k, 0, i])
    ri = C[k, 1, i]
    rj = C[k, 1, j]
    t = ri * rj
    ss = S[k, 1]
    if forward:
        a = C[k, 4, j] + C[k, 9, i]
        b = C[k, 5, j] + C[k, 10, i]
        c = C[k, 6, j] + C[k, 11, i]
        sign = 1.0
    else:
        a = C[k, 4, i] + C[k, 9, j]
        b = C[k, 5, i] + C[k, 10, j]
        c = C[k, 6, i] + C[k, 11, j]
        sign = -1.0
    tails = a + gap * (b + gap * c)
    G = t * (t - 3.0 * gap + S[k, 5]) + S[k, 3] * gap * gap + S[k, 4] * gap + S[k, 6]
    middle = sign * (G * (C[k, 2, j] - C[k, 2, i]) + (3.0 * gap + 3.0 - t - S[k, 7]) * (ri * C[k, 3, j] - rj * C[k, 3, i]) + ss * (C[k, 7, i] - C[k, 7, j])) / 9.0
    if S[k, 2] != 0.0:
        if forward:
            weight = C[k, 14, i] * C[k, 15, j]
        else:
            weight = C[k, 14, j] * C[k, 15, i]
    else:
        weight = exp(-gap)
    return weight * (tails + middle)


@njit(cache=True)
def _m52opt_workspace(n, d):
    return (np.empty(n), np.empty((d, 16, n)), np.empty((d, 8)), np.empty(n), np.empty(n), np.empty((min(32, n), n)))


@njit(cache=True, fastmath=False)
def _m52opt_contract(I, C, S, D, diagonal, tile, active, Rinv, Rinv_y):
    """Fuse IR, alpha.T J alpha and trace(Rinv J); lower triangle only.

    Rinv is symmetric, as assumed by the original implementation. Each tile
    retains dimension-outer evaluation while bounding temporary memory.
    """
    n = len(I)
    IR = 0.0
    tr = 0.0
    quad = 0.0
    for i in range(n):
        yi = Rinv_y[i]
        IR += I[i] * yi
        tr += Rinv[i, i] * diagonal[i]
        quad += diagonal[i] * yi * yi
    
    if active == 0:
        for i in range(n):
            for j in range(i):
                tr += 2.0 * Rinv[i, j] * I[i] * I[j]
        return IR, IR * IR, tr
    if active == 1:
        for i in range(n):
            yi = Rinv_y[i]
            for j in range(i):
                Jij = D[i] * D[j] * _m52opt_pair_compressed(C, S, i, j, 0)
                tr += 2.0 * Rinv[i, j] * Jij
                quad += 2.0 * Jij * yi * Rinv_y[j]
        return IR, quad, tr
    block = tile.shape[0]
    if n == 0:
        return IR, quad, tr
    for ib in range(0, n, block):
        ie = min(n, ib + block)
        for i in range(ib, ie):
            for j in range(i):
                tile[i - ib, j] = D[i] * D[j]
        for k in range(active):
            for i in range(ib, ie):
                for j in range(i):
                    tile[i - ib, j] *= _m52opt_pair_compressed(C, S, i, j, k)
        for i in range(ib, ie):
            yi = Rinv_y[i]
            for j in range(i):
                Jij = tile[i - ib, j]
                tr += 2.0 * Rinv[i, j] * Jij
                quad += 2.0 * Jij * yi * Rinv_y[j]
    return IR, quad, tr


@njit(cache=True, fastmath=False)
def _m52opt_query(m_row, v_row, w1, Rinv, Rinv_y, length_w, workspace):
    I, C, S, D, diagonal, tile = workspace
    active = _m52opt_prepare(w1, m_row, v_row, length_w, I, C, S, D, diagonal, True)
    _m52opt_compress(C, S, active)
    return _m52opt_contract(I, C, S, D, diagonal, tile, active, Rinv, Rinv_y)


@njit(cache=True, fastmath=False)
def _m52opt_query_withz(m_row, v_row, z_row, w1, global_w1, Rinv, Rinv_y,
                      length_w, inv_len_z, workspace):
    I, C, S, D, diagonal, tile = workspace
    active = _m52opt_prepare(w1, m_row, v_row, length_w, I, C, S, D, diagonal, True)
    _m52opt_compress(C, S, active)
    
    Izi = _matern25_kvec_invlen(global_w1, z_row, inv_len_z)
    for i in range(len(I)):
        I[i] *= Izi[i]
        D[i] *= Izi[i]
        diagonal[i] *= Izi[i] * Izi[i]
    return _m52opt_contract(I, C, S, D, diagonal, tile, active, Rinv, Rinv_y)


@njit(cache=True, fastmath=True, inline='always')
def _m52opt_tail_coefficients(r, s, ss, density):
    f0, f1, f2, f3, f4 = _m52opt_tail(r, s, ss, density)
    A = f0 + 2.0*f1 + (5.0/3.0)*f2 + (2.0/3.0)*f3 + f4/9.0
    B = f0 + (5.0/3.0)*f1 + f2 + (2.0/9.0)*f3
    C = (f0 + f1)/3.0 + f2/9.0
    return A, B, C





@njit(cache=True, fastmath=True, inline='always')
def _matern25_I1d(x, mu, var, ell, inv_ell, inv_ell2):
    """Exact compact first moment, including the deterministic limit."""
    if var == 0.0:
        return _matern25_det_1d(abs(mu - x) * inv_ell)
    lam = SQRT5 * inv_ell
    r = lam * (mu - x)
    ss = lam * lam * var
    s = sqrt(ss)
    density = exp(-0.5*r*r/ss) * INV_SQRT2PI
    return _m52opt_xi(r, s, ss, density)


@njit(cache=True, fastmath=True)
def Jd_nb(X1, X2, z_m, z_v, ell, ell2, ell3, ell4, inv_ell, inv_ell2):
    """M52OPT-05: compact scalar J; batched predictions use endpoint caching."""
    if z_v == 0.0:
        return (_matern25_det_1d(abs(z_m-X1)*inv_ell)
                * _matern25_det_1d(abs(z_m-X2)*inv_ell))
    lam = SQRT5 * inv_ell
    a = min(X1, X2)
    b = max(X1, X2)
    gap = lam * (b-a)
    ra = lam * (z_m-a)
    rb = lam * (z_m-b)
    ss = lam * lam * z_v
    s = sqrt(ss)
    da = exp(-0.5*ra*ra/ss) * INV_SQRT2PI
    db = exp(-0.5*rb*rb/ss) * INV_SQRT2PI
    Ap, Bp, Cp = _m52opt_tail_coefficients(rb, s, ss, db)
    Am, Bm, Cm = _m52opt_tail_coefficients(-ra, s, ss, da)
    tails = Ap + Am + gap*(Bp+Bm) + gap*gap*(Cp+Cm)
    Pa = s * da
    Pb = s * db
    deltaH = 0.5*(erf(-rb*INV_SQRT2/s)-erf(-ra*INV_SQRT2/s))
    t = ra * rb
    G = (t*(t-3.0*gap+6.0*ss-3.0) + (3.0+ss)*gap*gap
         + (9.0-3.0*ss)*gap + 9.0-3.0*ss+3.0*ss*ss)
    middle = (G*deltaH + (3.0*gap+3.0-t-4.0*ss)*(ra*Pb-rb*Pa)
              + ss*(ra*Pa-rb*Pb))/9.0
    return exp(-gap)*(tails+middle)



@njit(cache=True, fastmath=True)
def Jd0_nb(x1, z_m, z_v, ell, ell2, ell3, ell4, inv_ell, inv_ell2):
    """M52OPT-05: compact diagonal J, including exactly zero variance."""
    if z_v == 0.0:
        value = _matern25_det_1d(abs(z_m-x1)*inv_ell)
        return value * value
    lam = SQRT5 * inv_ell
    r = lam * (z_m-x1)
    ss = lam * lam * z_v
    s = sqrt(ss)
    density = exp(-0.5*r*r/ss) * INV_SQRT2PI
    Ap, Bp, Cp = _m52opt_tail_coefficients(r, s, ss, density)
    Am, Bm, Cm = _m52opt_tail_coefficients(-r, s, ss, density)
    return Ap + Am

@njit(cache=True, fastmath=True, inline="always")
def _Jii_matern25(w1, i, m_row, v_row,
                  length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w):
    """
    Compute J_ii without materialising J.
    """
    d = m_row.shape[0]
    out = 1.0
    for k in range(d):
        vk = v_row[k]
        if vk == 0.0:
            Iki = _matern25_det_1d(abs(m_row[k] - w1[i, k]) * inv_len_w[k])
            out *= (Iki * Iki)
        else:
            out *= Jd0_nb(w1[i, k], m_row[k], vk,
                       length_w[k], len2_w[k], len3_w[k], len4_w[k],
                       inv_len_w[k], inv_len2_w[k]) 
    return out


@njit(cache=True, fastmath=True, inline="always")
def _Jij_matern25(w1, i, j, m_row, v_row,
                  length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w):
    """
    Compute J_ij (i!=j) without materialising J.
    """
    d = m_row.shape[0]
    out = 1.0
    for k in range(d):
        vk = v_row[k]
        if vk == 0.0:
            Iki = _matern25_det_1d(abs(m_row[k] - w1[i, k]) * inv_len_w[k])
            Ikj = _matern25_det_1d(abs(m_row[k] - w1[j, k]) * inv_len_w[k])
            out *= (Iki * Ikj)
        else:
            out *= Jd_nb(w1[j, k], w1[i, k], m_row[k], vk,
                      length_w[k], len2_w[k], len3_w[k], len4_w[k],
                      inv_len_w[k], inv_len2_w[k])  
    return out


# ============================================================
# matern2.5 fast: z is None (4 functions style like sexp)
# ============================================================


@njit(cache=True, fastmath=True)
def link_gp_matern25_noz_serial(m, v, w1, Rinv, Rinv_y, scale, length_w, nugget):
    """Cached exact Matérn-5/2 prediction; no full J allocation."""
    n_pred, d = m.shape
    workspace = _m52opt_workspace(w1.shape[0], d)
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    for t in range(n_pred):
        IR, quad, tr = _m52opt_query(m[t], v[t], w1, Rinv, Rinv_y, length_w, workspace)
        m_new[t] = IR
        
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))
    return m_new, v_new



@njit(cache=True, parallel=True, fastmath=True)
def link_gp_matern25_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, length_w, nugget):
    """Cached exact Matérn-5/2 prediction, parallel over independent queries."""
    n_pred, d = m.shape
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    for t in prange(n_pred):
        workspace = _m52opt_workspace(w1.shape[0], d)
        IR, quad, tr = _m52opt_query(m[t], v[t], w1, Rinv, Rinv_y, length_w, workspace)
        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))
    return m_new, v_new


# ============================================================
# matern2.5 fast: z is not None (4 functions style like sexp)
# ============================================================


@njit(cache=True, fastmath=True)
def link_gp_matern25_withz_serial(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget):
    """M52OPT-04/06: same API; cached moments and deterministic global factors."""
    n_pred, d = m.shape
    workspace = _m52opt_workspace(w1.shape[0], d)
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    for t in range(n_pred):
        IR, quad, tr = _m52opt_query_withz(m[t], v[t], z[t], w1, global_w1,
                                         Rinv, Rinv_y, length_w, inv_len_z, workspace)
        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))
    return m_new, v_new



@njit(cache=True, parallel=True, fastmath=True)
def link_gp_matern25_withz_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget):
    """M52OPT-04/07: same global-input API; private caches inside prange."""
    n_pred, d = m.shape
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)
    for t in prange(n_pred):
        workspace = _m52opt_workspace(w1.shape[0], d)
        IR, quad, tr = _m52opt_query_withz(m[t], v[t], z[t], w1, global_w1,
                                         Rinv, Rinv_y, length_w, inv_len_z, workspace)
        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))
    return m_new, v_new


from math import log, log1p, isfinite

@njit(cache=True, fastmath=False, inline='never')
def _m5_cost(t):
    """Evaluate t-log(1+t+t*t/3) without dangerous intermediates."""
    if not isfinite(t):
        return t
    if t <= 0.001:
        t2 = t * t
        return t2 * (1.0 / 6.0 + t2 * (-1.0 / 36.0 + t * (1.0 / 45.0 - t / 81.0)))
    if t <= 1.0:
        return t - log1p(t + t * t / 3.0)
    u = 1.0 / t
    return t - 2.0 * log(t) + log(3.0) - log1p(3.0 * u + 3.0 * u * u)

@njit(cache=True, fastmath=False, inline='never')
def _m5_log_pair(x, z, lengths, mode):
    """mode: 0 already scaled; 1 inverse lengths; 2 original lengths."""
    total = 0.0
    correction = 0.0
    for k in range(x.size):
        delta = x[k] - z[k]
        if mode == 0:
            r = abs(delta)
        else:
            ell = lengths[0] if lengths.size == 1 else lengths[k]
            if mode == 1:
                r = abs(delta * ell) if isfinite(delta) else abs(x[k] * ell - z[k] * ell)
            elif isfinite(delta):
                r = abs(delta / ell)
            else:
                r = abs(x[k] / ell - z[k] / ell)
        value = _m5_cost(SQRT5 * r)
        if not isfinite(value):
            return exp(-value)
        adjusted = value - correction
        new_total = total + adjusted
        if new_total > 746.0:
            return 0.0
        correction = new_total - total - adjusted
        total = new_total
    return exp(-total)


def _sum_count_loglik(values):
    """Allow a negative total outside float64 range to reduce to -inf."""
    if np.all(values <= 0):
        with np.errstate(over='ignore'):
            return np.sum(values)
    return np.sum(values)


def _zi_logpmf(y, base, logit):
    """Add the structural-zero mass only at zero observations."""
    y, base, logit = np.broadcast_arrays(y, base, np.asarray(logit, dtype=float))
    weighted = np.asarray(-np.logaddexp(0., logit) + base)
    zero = y == 0
    weighted[zero] = np.logaddexp(-np.logaddexp(0., -logit[zero]), weighted[zero])
    return weighted


def _zip_logpmf(y, log_rate, logit):
    observed = np.asarray(y)
    y, log_rate = np.broadcast_arrays(observed, np.asarray(log_rate, dtype=float))
    with np.errstate(over='ignore'):
        base = np.asarray(-np.exp(log_rate))
    positive = (y != 0) & np.isfinite(base)
    if observed.shape == y.shape:
        log_factorial = gammaln(y[positive] + 1.)
    else:
        log_factorial = np.broadcast_to(gammaln(observed + 1.), y.shape)[positive]
    base[positive] += y[positive] * log_rate[positive] - log_factorial
    return _zi_logpmf(y, base, logit)


def _nb_direct(y, log_mean, log_disp, log_factorial):
    size = np.exp(-log_disp)
    a = log_mean + log_disp
    softplus = np.logaddexp(0., a)
    return (gammaln(y + size) - gammaln(size) - log_factorial
            + y * a - (y + size) * softplus)


def _nb_size_penalty(log_mean, log_disp, a):
    """Compute size * log(1 + exp(a)) even when size overflows."""
    low = a <= 0
    with np.errstate(over='ignore', under='ignore'):
        scale = np.exp(.5 * np.where(low, log_mean, -log_disp))
        weight = np.logaddexp(0., a)
        z = np.exp(a[low])
        weight[low] = np.divide(np.log1p(z), z, out=np.ones_like(z), where=z != 0)
        return (scale * weight) * scale


def _nb_gamma_correction(increment, log_increment, log_base):
    """logGamma(base+h) - logGamma(base) - h*log(base), for base >= 1024.

    Expand ((1+t)*log1p(t)-t)/t near zero and difference the first two
    Stirling corrections directly to retain precision when h/base is small.
    """
    reciprocal = np.exp(-log_base)
    ratio = np.exp(log_increment - log_base)
    log_ratio = np.log1p(ratio)
    small = ratio <= .01
    t = ratio[small]
    coefficients = (.5, -1/6, 1/12, -1/20, 1/30, -1/42, 1/56)
    quotient = np.empty_like(ratio)
    quotient[small] = t * np.polynomial.polynomial.polyval(t, coefficients)
    quotient[~small] = (1 + 1/ratio[~small]) * log_ratio[~small] - 1
    return (increment * quotient - .5 * log_ratio + reciprocal/12 * np.expm1(-log_ratio)
            - reciprocal**3/360 * np.expm1(-3 * log_ratio))


def _nb_tail(y, log_mean, log_disp, log_factorial):
    """Use a large size or count as the gamma-ratio base. With both small,
    shift the beta coefficient by one so size may underflow to zero.
    """
    a = log_mean + log_disp
    result = -_nb_size_penalty(log_mean, log_disp, a)
    large_size = log_disp <= -np.log(1024.)
    large = (y > 0) & large_size
    if np.any(large):
        count, disp, x = y[large], log_disp[large], a[large]
        log_scaled_mean = np.where(x <= 0, log_mean[large] - np.logaddexp(0., x),
                                   -disp - np.logaddexp(0., -x))
        result[large] += (_nb_gamma_correction(count, np.log(count), -disp) + count * log_scaled_mean
                          - log_factorial[large])
    small = (y > 0) & ~large_size
    if np.any(small):
        count, disp = y[small], log_disp[small]
        size = np.exp(-disp)
        log_count = np.log(count)
        coefficient = -disp - log_count
        high = count >= 1024
        if np.any(high):
            n, log_y = size[high], log_count[high]
            correction = _nb_gamma_correction(n, -disp[high], log_y)
            coefficient[high] += n * log_y + correction - gammaln(1 + n)
        if np.any(~high):
            y_low, n_low = count[~high], size[~high]
            coefficient[~high] -= np.log(y_low + n_low) + betaln(y_low, 1 + n_low)
        result[small] += coefficient - count * np.logaddexp(0., -a[small])
    return result


def _nb_logpmf(y, log_mean, log_disp):
    """Use the direct formula on ordinary inputs and on whole invalid batches."""
    observed = np.asarray(y, dtype=float)
    y, log_mean, log_disp = np.broadcast_arrays(observed, np.asarray(log_mean, dtype=float),
                                                np.asarray(log_disp, dtype=float))
    log_factorial = gammaln(observed + 1.)
    ordinary = (y <= 100) & (np.abs(log_mean) <= 20) & (np.abs(log_disp) <= np.log(1024.))
    if (np.all(ordinary)
            or not np.all(np.isfinite(log_mean)) or not np.all(np.isfinite(log_disp))
            or not np.all(np.isfinite(observed) & (observed >= 0) & (observed == np.floor(observed)))):
        return _nb_direct(y, log_mean, log_disp, log_factorial)
    log_factorial = np.broadcast_to(log_factorial, y.shape)
    result = np.empty(y.shape)
    if np.any(ordinary):
        result[ordinary] = _nb_direct(y[ordinary], log_mean[ordinary], log_disp[ordinary],
                                      log_factorial[ordinary])
    tail = ~ordinary
    result[tail] = _nb_tail(y[tail], log_mean[tail], log_disp[tail], log_factorial[tail])
    return result
