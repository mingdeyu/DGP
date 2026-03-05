from numba import njit, prange, config, set_num_threads
import numpy as np
from math import erf, sqrt, pi, exp, fabs
from numpy.random import randn
from scipy.linalg import pinvh, cholesky, solve_triangular, LinAlgError
import itertools
from psutil import cpu_count
from functools import lru_cache

core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num, max_threads)
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

@lru_cache(maxsize=None)
def _hermgauss_1d(q: int):
    x, w = np.polynomial.hermite.hermgauss(q)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)

@lru_cache(maxsize=None)
def _gh_tensor_grid(N: int, q: int):
    """
    Tensor-product Gauss–Hermite nodes/weights in the SAME ordering as itertools.product(x, repeat=N),
    but built in NumPy (much faster) and cached.
    """
    x, w = _hermgauss_1d(q)

    X = np.meshgrid(*([x] * N), indexing="ij")
    xn = np.stack(X, axis=-1).reshape(-1, N)  # (P, N)

    W = np.meshgrid(*([w] * N), indexing="ij")
    wn = np.prod(np.stack(W, axis=-1), axis=-1).reshape(-1)  # (P,)

    logw = np.log(wn) - 0.5 * N * np.log(np.pi)  # log(wn * pi^{-N/2})
    return xn, logw

def _logsumexp_axis1(a: np.ndarray) -> np.ndarray:
    """Compute logsumexp over axis=1 using only NumPy. a shape (M, K)."""
    amax = np.max(a, axis=1, keepdims=True)
    # handle all -inf rows safely
    with np.errstate(under="ignore"):
        s = np.sum(np.exp(a - amax), axis=1)
    return amax[:, 0] + np.log(s)

def ghdiag(fct, mu, var, y, q: int = 10, block: int = 8192, max_nodes: int = 2_000_000):
    """
    Gauss–Hermite quadrature for E[ exp(loglik(y | f)) ] with diagonal Gaussian uncertainty:
      f ~ N(mu, var) elementwise across N dims.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    y = np.asarray(y)

    M, N = mu.shape
    P = q ** N
    if P > max_nodes:
        raise ValueError(
            f"Tensor GH grid too large: q**N = {q}^{N} = {P} nodes. "
            f"Increase max_nodes, reduce N/q, or switch to a different approximation."
        )

    xn, logw = _gh_tensor_grid(N, q)  # (P,N), (P,)
    sqrt_var = np.sqrt(var)
    sqrt2 = np.sqrt(2.0)

    # accumulate log(sum over all nodes) per data point, in log-space across chunks
    log_total = np.full((M,), -np.inf, dtype=np.float64)

    for j in range(0, P, block):
        sl = slice(j, min(j + block, P))
        xn_b = xn[sl]          # (B, N)
        logw_b = logw[sl]      # (B,)

        fn_b = sqrt2 * (sqrt_var[:, None, :] * xn_b[None, :, :]) + mu[:, None, :]

        llik = np.asarray(fct(y[:, None], fn_b))
        if llik.ndim == 3 and llik.shape[-1] == 1:
            llik = llik[..., 0]     # (M, B)
        elif llik.ndim != 2:
            raise ValueError(f"Unexpected llik shape {llik.shape}; expected (M,B) or (M,B,1).")

        # log(sum_{nodes in block} exp(llik + logw))
        log_block = _logsumexp_axis1(llik + logw_b[None, :])

        # combine blocks: log_total = logaddexp(log_total, log_block)
        log_total = np.logaddexp(log_total, log_block)

    return np.exp(log_total)

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

@njit(cache=True)
def K_matern25_nb(xi, length, nugget):
    n, d = xi.shape
    iso = (length.size == 1)

    K = np.empty((n, n), dtype=np.float64)
    nug_scalar = (nugget.size == 1)

    #SQRT5 = 2.2360679774997898
    #FIVE_THIRDS = 1.6666666666666667  # 5/3

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

    #SQRT5 = 2.2360679774997898
    #FIVE_THIRDS = 1.6666666666666667  # 5/3

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

    #SQRT5 = 2.2360679774997898                                   
    #FIVE_THIRDS = 1.6666666666666667

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

@njit(cache=True, parallel=True)
def gp(x,w1,Rinv,Rinv_y,scale,nugget,name):
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

@njit(cache=True, fastmath=True)
def link_gp_sexp_noz_serial(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget):
    """
    sexp linked GP without forming Ii/Ji, z is None.
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    denI = np.empty(d, dtype=np.float64)
    denJ = np.empty(d, dtype=np.float64)

    for t in range(n_pred):
        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv = inv_len_w[k]
            ms[k] = m[t, k] * inv
            vs[k] = v[t, k] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        quad = 0.0
        tr = 0.0
        IR = 0.0

        for i in range(n):
            expo_i = 0.0
            for k in range(d):
                xi = w1[i, k] * inv_len_w[k]
                diff = xi - ms[k]
                expo_i += diff * diff * denI[k]
            Ii = I_coef1 * exp(-expo_i)
            IR += Ii * Rinv_y[i]

            yi = Rinv_y[i]
            for j in range(i + 1):
                yj = Rinv_y[j]
                Jij = _Jij_sexp_scaled(w1, i, j, ms, denJ, inv_len_w, J_coef1)

                Rij = Rinv[i, j]
                if i == j:
                    tr += Rij * Jij
                    quad += Jij * yi * yi
                else:
                    tr += 2.0 * Rij * Jij
                    quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=True, parallel=True, fastmath=True)
def link_gp_sexp_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget):
    """Same as serial but prange over prediction points."""
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    for t in prange(n_pred):
        denI = np.empty(d, dtype=np.float64)     
        denJ = np.empty(d, dtype=np.float64)   

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv = inv_len_w[k]
            ms[k] = m[t, k] * inv
            vs[k] = v[t, k] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        quad = 0.0
        tr = 0.0
        IR = 0.0

        for i in range(n):
            expo_i = 0.0
            for k in range(d):
                xi = w1[i, k] * inv_len_w[k]
                diff = xi - ms[k]
                expo_i += diff * diff * denI[k]
            Ii = I_coef1 * exp(-expo_i)
            IR += Ii * Rinv_y[i]

            yi = Rinv_y[i]
            for j in range(i + 1):
                yj = Rinv_y[j]
                Jij = _Jij_sexp_scaled(w1, i, j, ms, denJ, inv_len_w, J_coef1)

                Rij = Rinv[i, j]
                if i == j:
                    tr += Rij * Jij
                    quad += Jij * yi * yi
                else:
                    tr += 2.0 * Rij * Jij
                    quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

@njit(cache=True, fastmath=True)
def link_gp_sexp_withz_serial(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget):
    """
    sexp linked GP with global z, without forming Ji or outer(Izi,Izi).
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    denI = np.empty(d, dtype=np.float64)
    denJ = np.empty(d, dtype=np.float64)

    for t in range(n_pred):
        Izi = _sexp_kvec_invlen(global_w1, z[t], inv_len_z)
        y_scaled = Rinv_y * Izi

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv = inv_len_w[k]
            ms[k] = m[t, k] * inv
            vs[k] = v[t, k] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        quad = 0.0
        tr = 0.0
        IR = 0.0

        for i in range(n):
            expo_i = 0.0
            for k in range(d):
                xi = w1[i, k] * inv_len_w[k]
                diff = xi - ms[k]
                expo_i += diff * diff * denI[k]
            Ii = I_coef1 * exp(-expo_i)
            IR += Ii * y_scaled[i]

            yi = y_scaled[i]
            si = Izi[i]
            for j in range(i + 1):
                yj = y_scaled[j]
                sj = Izi[j]
                Jij = _Jij_sexp_scaled(w1, i, j, ms, denJ, inv_len_w, J_coef1)

                Rij = Rinv[i, j] * (si * sj)

                if i == j:
                    tr += Rij * Jij
                    quad += Jij * yi * yi
                else:
                    tr += 2.0 * Rij * Jij
                    quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

@njit(cache=True, parallel=True, fastmath=True)
def link_gp_sexp_withz_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget):
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    for t in prange(n_pred):
        denI = np.empty(d, dtype=np.float64)
        denJ = np.empty(d, dtype=np.float64)

        Izi = _sexp_kvec_invlen(global_w1, z[t], inv_len_z)
        y_scaled = Rinv_y * Izi

        ms = np.empty(d, dtype=np.float64)
        vs = np.empty(d, dtype=np.float64)
        for k in range(d):
            inv = inv_len_w[k]
            ms[k] = m[t, k] * inv
            vs[k] = v[t, k] * (inv * inv)

        I_coef1, J_coef1 = _compute_denoms_from_vs_scaled(vs, denI, denJ)

        quad = 0.0
        tr = 0.0
        IR = 0.0

        for i in range(n):
            expo_i = 0.0
            for k in range(d):
                xi = w1[i, k] * inv_len_w[k]
                diff = xi - ms[k]
                expo_i += diff * diff * denI[k]
            Ii = I_coef1 * exp(-expo_i)
            IR += Ii * y_scaled[i]

            yi = y_scaled[i]
            si = Izi[i]
            for j in range(i + 1):
                yj = y_scaled[j]
                sj = Izi[j]
                Jij = _Jij_sexp_scaled(w1, i, j, ms, denJ, inv_len_w, J_coef1)

                Rij = Rinv[i, j] * (si * sj)

                if i == j:
                    tr += Rij * Jij
                    quad += Jij * yi * yi
                else:
                    tr += 2.0 * Rij * Jij
                    quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new

# linked GP predictions (non-vecchia, matern2.5)

@njit(cache=True, fastmath=True, inline="always")
def _Phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + erf(x * INV_SQRT2))


@njit(cache=True, fastmath=True, inline="always")
def _matern25_det_1d(absdiff_scaled):
    """
    Deterministic matern2.5 kernel in 1D with r = |x-mu|/ell already scaled:
      (1 + sqrt(5) r + 5 r^2/3) * exp(-sqrt(5) r)
    """
    r = absdiff_scaled
    poly = 1.0 + SQRT5 * r + FIVE_THIRDS * r * r
    return poly * exp(-SQRT5 * r)


@njit(cache=True, fastmath=True, inline="always")
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
            coef1 *= (1.0 + SQRT5 * distk + FIVE_THIRDS * distk * distk)
            coef2 += distk
        out[i] = coef1 * exp(-SQRT5 * coef2)
    return out


@njit(cache=True, fastmath=True, inline="always")
def _matern25_I1d(x, mu, var, ell, inv_ell, inv_ell2):
    """
    1D factor xi_{ik} (IJ_matern 'Ii' per-dimension piece).
    """
    if var == 0.0:
        r = abs(mu - x) * inv_ell
        return _matern25_det_1d(r)

    zX = mu - x
    a = SQRT5 * zX * inv_ell

    # common factor exp((5*var)/(2*ell^2))
    base = exp(2.5 * var * inv_ell2)

    # muA = zX - sqrt5*var/ell ; muB = zX + sqrt5*var/ell
    s = SQRT5 * var * inv_ell
    muA = zX - s
    muB = zX + s

    inv_sqrt_var = 1.0 / sqrt(var)

    PhiA = _Phi(muA * inv_sqrt_var)
    PhiB = _Phi(-muB * inv_sqrt_var)

    # exp(-0.5*mu^2/var) parts
    phiA = exp(-0.5 * (muA * muA) / var)
    phiB = exp(-0.5 * (muB * muB) / var)

    # sqrt(var/(2*pi))/ell  == sqrt(var) * (1/sqrt(2pi)) / ell
    norm = (sqrt(var) * INV_SQRT2PI) * inv_ell

    # polynomial pieces
    A1 = 1.0 + SQRT5 * muA * inv_ell + FIVE_THIRDS * (muA * muA + var) * inv_ell2
    A2 = SQRT5 + FIVE_THIRDS * muA * inv_ell
    partA = A1 * PhiA + A2 * norm * phiA

    B1 = 1.0 - SQRT5 * muB * inv_ell + FIVE_THIRDS * (muB * muB + var) * inv_ell2
    B2 = SQRT5 - FIVE_THIRDS * muB * inv_ell
    partB = B1 * PhiB + B2 * norm * phiB

    # exp factors exp(±a)
    return base * (exp(-a) * partA + exp(a) * partB)

@njit(cache=True, fastmath=True)
def Jd_nb(X1, X2, z_m, z_v, ell, ell2, ell3, ell4, inv_ell, inv_ell2):
    """Compute J components in 1D for Matern2.5 kernel. (SAME math)"""
    if X1 < X2:
        x1 = X1
        x2 = X2
    else:
        x1 = X2
        x2 = X1

    # ===== common scalars =====
    den = 1.0 / (9.0 * ell4)                   
    sigma = sqrt(z_v)                       
    inv_sigma = 1.0 / sigma                    
    inv_sqrt_2zv = 1.0 / sqrt(2.0 * z_v)    
    pdf_scale = sqrt(0.5 * z_v / pi)   

    S = x1 + x2                                 
    D = x2 - x1                                  
    P = x1 * x2                                  
    x1_sq = x1 * x1                               
    x2_sq = x2 * x2                               
    x2_cu = x2_sq * x2                            

    base10 = 10.0 * z_v * inv_ell2                
    shiftS = SQRT5 * (S - 2.0 * z_m) * inv_ell    

    # ---- E3 block ----
    E30 = 1.0 + (25.0 * x1_sq * x2_sq
                 - 3.0 * SQRT5 * (3.0 * ell3 + 5.0 * ell * P) * S
                 + 15.0 * ell2 * (x1_sq + x2_sq + 3.0 * P)) * den

    E31 = (18.0 * SQRT5 * ell3
           + 15.0 * SQRT5 * ell * (x1_sq + x2_sq)
           - (75.0 * ell2 + 50.0 * P) * S
           + 60.0 * SQRT5 * ell * P) * den

    E32 = 5.0 * (5.0 * x1_sq + 5.0 * x2_sq + 15.0 * ell2
                 - 9.0 * SQRT5 * ell * S
                 + 20.0 * P) * den

    E33 = 10.0 * (3.0 * SQRT5 * ell - 5.0 * x1 - 5.0 * x2) * den
    E34 = 25.0 * den

    muC = z_m - 2.0 * SQRT5 * z_v * inv_ell  

    muC2 = muC * muC
    muC3 = muC2 * muC
    muC4 = muC2 * muC2

    E3A31 = (E30
             + muC * E31
             + (muC2 + z_v) * E32
             + (muC3 + 3.0 * z_v * muC) * E33
             + (muC4 + 6.0 * z_v * muC2 + 3.0 * z_v * z_v) * E34)

    E3A32 = (E31
             + (muC + x2) * E32
             + (muC2 + 2.0 * z_v + x2_sq + muC * x2) * E33
             + (muC3 + x2_cu + x2 * muC2 + muC * x2_sq + 3.0 * z_v * x2 + 5.0 * z_v * muC) * E34)

    exp1 = exp(base10 + shiftS)  
    uC = (muC - x2) * inv_sqrt_2zv    
    pdfC = pdf_scale * exp(-0.5 * ((x2 - muC) * inv_sigma) ** 2) 

    P1 = exp1 * (
        0.5 * E3A31 * (1.0 + erf(uC)) +
        E3A32 * pdfC
    )

    # ---- E4 block ----
    E40 = 1.0 + (25.0 * x1_sq * x2_sq
                 + 3.0 * SQRT5 * (3.0 * ell3 - 5.0 * ell * P) * D
                 + 15.0 * ell2 * (x1_sq + x2_sq - 3.0 * P)) * den

    E41 = 5.0 * (3.0 * SQRT5 * ell * (x2_sq - x1_sq)
                 + 3.0 * ell2 * S
                 - 10.0 * P * S) * den

    E42 = 5.0 * (5.0 * x1_sq + 5.0 * x2_sq - 3.0 * ell2
                 - 3.0 * SQRT5 * ell * D
                 + 20.0 * P) * den

    E43 = -50.0 * S * den                 
    E44 = 25.0 * den

    zm2 = z_m * z_m
    zm3 = zm2 * z_m
    zm4 = zm2 * zm2

    E4A41 = (E40
             + z_m * E41
             + (zm2 + z_v) * E42
             + (zm3 + 3.0 * z_v * z_m) * E43
             + (zm4 + 6.0 * z_v * zm2 + 3.0 * z_v * z_v) * E44)

    E4A42 = (E41
             + (z_m + x1) * E42
             + (zm2 + 2.0 * z_v + x1_sq + z_m * x1) * E43
             + (zm3 + (x1_sq * x1) + x1 * zm2 + z_m * x1_sq + 3.0 * z_v * x1 + 5.0 * z_v * z_m) * E44)

    E4A43 = (E41
             + (z_m + x2) * E42
             + (zm2 + 2.0 * z_v + x2_sq + z_m * x2) * E43
             + (zm3 + x2_cu + x2 * zm2 + z_m * x2_sq + 3.0 * z_v * x2 + 5.0 * z_v * z_m) * E44)

    exp2 = exp(-SQRT5 * D * inv_ell)  

    u2 = (x2 - z_m) * inv_sqrt_2zv        
    u1 = (x1 - z_m) * inv_sqrt_2zv        

    pdf1 = pdf_scale * exp(-0.5 * ((x1 - z_m) * inv_sigma) ** 2)  
    pdf2 = pdf_scale * exp(-0.5 * ((x2 - z_m) * inv_sigma) ** 2) 

    P2 = exp2 * (
        0.5 * E4A41 * (erf(u2) - erf(u1)) +
        E4A42 * pdf1 -
        E4A43 * pdf2
    )

    # ---- E5 block ----
    E50 = 1.0 + (25.0 * x1_sq * x2_sq
                 + 3.0 * SQRT5 * (3.0 * ell3 + 5.0 * ell * P) * S
                 + 15.0 * ell2 * (x1_sq + x2_sq + 3.0 * P)) * den

    E51 = (18.0 * SQRT5 * ell3
           + 15.0 * SQRT5 * ell * (x1_sq + x2_sq)
           + (75.0 * ell2 + 50.0 * P) * S
           + 60.0 * SQRT5 * ell * P) * den

    E52 = 5.0 * (5.0 * x1_sq + 5.0 * x2_sq + 15.0 * ell2
                 + 9.0 * SQRT5 * ell * S
                 + 20.0 * P) * den

    E53 = 10.0 * (3.0 * SQRT5 * ell + 5.0 * x1 + 5.0 * x2) * den
    E54 = 25.0 * den

    muD = z_m + 2.0 * SQRT5 * z_v * inv_ell  

    muD2 = muD * muD
    muD3 = muD2 * muD
    muD4 = muD2 * muD2

    E5A51 = (E50
             - muD * E51
             + (muD2 + z_v) * E52
             - (muD3 + 3.0 * z_v * muD) * E53
             + (muD4 + 6.0 * z_v * muD2 + 3.0 * z_v * z_v) * E54)

    E5A52 = (E51
             - (muD + x1) * E52
             + (muD2 + 2.0 * z_v + x1_sq + muD * x1) * E53
             - (muD3 + (x1_sq * x1) + x1 * muD2 + muD * x1_sq + 3.0 * z_v * x1 + 5.0 * z_v * muD) * E54)

    exp3 = exp(base10 - shiftS)  

    uD = (x1 - muD) * inv_sqrt_2zv
    pdfD = pdf_scale * exp(-0.5 * ((x1 - muD) * inv_sigma) ** 2)

    P3 = exp3 * (
        0.5 * E5A51 * (1.0 + erf(uD)) +
        E5A52 * pdfD
    )

    return P1 + P2 + P3


@njit(cache=True, fastmath=True)
def Jd0_nb(x1, z_m, z_v, ell, ell2, ell3, ell4, inv_ell, inv_ell2):
    """Diagonal J component in 1D for Matern2.5."""

    den = 1.0 / (9.0 * ell4)                      
    sigma = sqrt(z_v)                       
    inv_sigma = 1.0 / sigma                     
    inv_sqrt_2zv = 1.0 / sqrt(2.0 * z_v)   
    pdf_scale = sqrt(0.5 * z_v / pi)   

    x1_sq = x1 * x1                              
    x1_4 = x1_sq * x1_sq                         

    base10 = 10.0 * z_v * inv_ell2                
    shift2 = SQRT5 * (2.0 * x1 - 2.0 * z_m) * inv_ell  

    # ---- E3 block ----
    E30 = 1.0 + (25.0 * x1_4
                 - 6.0 * SQRT5 * (3.0 * ell3 + 5.0 * ell * x1_sq) * x1
                 + 75.0 * ell2 * x1_sq) * den

    E31 = (18.0 * SQRT5 * ell3
           + 90.0 * SQRT5 * ell * x1_sq
           - (150.0 * ell2 + 100.0 * x1_sq) * x1) * den

    E32 = 5.0 * (30.0 * x1_sq + 15.0 * ell2 - 18.0 * SQRT5 * ell * x1) * den
    E33 = 10.0 * (3.0 * SQRT5 * ell - 10.0 * x1) * den
    E34 = 25.0 * den

    muC = z_m - 2.0 * SQRT5 * z_v * inv_ell

    muC2 = muC * muC
    muC3 = muC2 * muC
    muC4 = muC2 * muC2

    E3A31 = (E30
             + muC * E31
             + (muC2 + z_v) * E32
             + (muC3 + 3.0 * z_v * muC) * E33
             + (muC4 + 6.0 * z_v * muC2 + 3.0 * z_v * z_v) * E34)

    E3A32 = (E31
             + (muC + x1) * E32
             + (muC2 + 2.0 * z_v + x1_sq + muC * x1) * E33
             + (muC3 + (x1_sq * x1) + x1 * muC2 + muC * x1_sq + 3.0 * z_v * x1 + 5.0 * z_v * muC) * E34)

    exp1 = exp(base10 + shift2)
    uC = (muC - x1) * inv_sqrt_2zv
    pdfC = pdf_scale * exp(-0.5 * ((x1 - muC) * inv_sigma) ** 2)

    P1 = exp1 * (
        0.5 * E3A31 * (1.0 + erf(uC)) +
        E3A32 * pdfC
    )

    # ---- E5 block ----
    E50 = 1.0 + (25.0 * x1_4
                 + 6.0 * SQRT5 * (3.0 * ell3 + 5.0 * ell * x1_sq) * x1
                 + 75.0 * ell2 * x1_sq) * den

    E51 = (18.0 * SQRT5 * ell3
           + 90.0 * SQRT5 * ell * x1_sq
           + (150.0 * ell2 + 100.0 * x1_sq) * x1) * den

    E52 = 5.0 * (30.0 * x1_sq + 15.0 * ell2 + 18.0 * SQRT5 * ell * x1) * den
    E53 = 10.0 * (3.0 * SQRT5 * ell + 10.0 * x1) * den
    E54 = 25.0 * den

    muD = z_m + 2.0 * SQRT5 * z_v * inv_ell

    muD2 = muD * muD
    muD3 = muD2 * muD
    muD4 = muD2 * muD2

    E5A51 = (E50
             - muD * E51
             + (muD2 + z_v) * E52
             - (muD3 + 3.0 * z_v * muD) * E53
             + (muD4 + 6.0 * z_v * muD2 + 3.0 * z_v * z_v) * E54)

    E5A52 = (E51
             - (muD + x1) * E52
             + (muD2 + 2.0 * z_v + x1_sq + muD * x1) * E53
             - (muD3 + (x1_sq * x1) + x1 * muD2 + muD * x1_sq + 3.0 * z_v * x1 + 5.0 * z_v * muD) * E54)

    exp3 = exp(base10 - shift2)
    uD = (x1 - muD) * inv_sqrt_2zv
    pdfD = pdf_scale * exp(-0.5 * ((x1 - muD) * inv_sigma) ** 2)

    P3 = exp3 * (
        0.5 * E5A51 * (1.0 + erf(uD)) +
        E5A52 * pdfD
    )

    return P1 + P3

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
    """
    Matérn-2.5 linked GP without forming I/J; z is None.
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in range(n_pred):
        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(n):
            # ---- IR part ----
            Ii = 1.0
            for k in range(d):
                Ii *= _matern25_I1d(w1[i, k], m_row[k], v_row[k],
                                    length_w[k], inv_len_w[k], inv_len2_w[k])
            yi = Rinv_y[i]
            IR += Ii * yi

            # ---- diagonal contribution ----
            Jii = _Jii_matern25(w1, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            tr += Rinv[i, i] * Jii
            quad += Jii * yi * yi

            # ---- off-diagonal (use symmetry) ----
            for j in range(i):
                Jij = _Jij_matern25(w1, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                Rij = Rinv[i, j]
                yj = Rinv_y[j]
                tr += 2.0 * Rij * Jij
                quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=True, parallel=True, fastmath=True)
def link_gp_matern25_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, length_w, nugget):
    """Same as serial but prange over prediction points."""
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in prange(n_pred):
        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(n):
            Ii = 1.0
            for k in range(d):
                Ii *= _matern25_I1d(w1[i, k], m_row[k], v_row[k],
                                    length_w[k], inv_len_w[k], inv_len2_w[k])
            yi = Rinv_y[i]
            IR += Ii * yi

            Jii = _Jii_matern25(w1, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            tr += Rinv[i, i] * Jii
            quad += Jii * yi * yi

            for j in range(i):
                Jij = _Jij_matern25(w1, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                Rij = Rinv[i, j]
                yj = Rinv_y[j]
                tr += 2.0 * Rij * Jij
                quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


# ============================================================
# matern2.5 fast: z is not None (4 functions style like sexp)
# ============================================================

@njit(cache=True, fastmath=True)
def link_gp_matern25_withz_serial(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget):
    """
    Matérn-2.5 linked GP with global z (deterministic Matérn for global part),
    without forming I/J or outer(Izi,Izi).
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in range(n_pred):
        Izi = _matern25_kvec_invlen(global_w1, z[t], inv_len_z)   # (n,)
        y_scaled = Rinv_y * Izi                                   # (n,)

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(n):
            Ii = 1.0
            for k in range(d):
                Ii *= _matern25_I1d(w1[i, k], m_row[k], v_row[k],
                                    length_w[k], inv_len_w[k], inv_len2_w[k])

            si = Izi[i]
            yi = y_scaled[i]
            IR += Ii * yi

            # diagonal
            Jii = _Jii_matern25(w1, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            Rii = Rinv[i, i] * (si * si)     
            tr += Rii * Jii
            quad += Jii * yi * yi

            # off-diagonal
            for j in range(i):
                Jij = _Jij_matern25(w1, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                sj = Izi[j]
                Rij = Rinv[i, j] * (si * sj)     
                yj = y_scaled[j]
                tr += 2.0 * Rij * Jij
                quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=True, parallel=True, fastmath=True)
def link_gp_matern25_withz_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget):
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    inv_len_w = 1.0 / length_w
    inv_len2_w = inv_len_w * inv_len_w
    len2_w = length_w * length_w
    len3_w = len2_w * length_w
    len4_w = len2_w * len2_w

    for t in prange(n_pred):
        Izi = _matern25_kvec_invlen(global_w1, z[t], inv_len_z)
        y_scaled = Rinv_y * Izi

        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(n):
            Ii = 1.0
            for k in range(d):
                Ii *= _matern25_I1d(w1[i, k], m_row[k], v_row[k],
                                    length_w[k], inv_len_w[k], inv_len2_w[k])

            si = Izi[i]
            yi = y_scaled[i]
            IR += Ii * yi

            Jii = _Jii_matern25(w1, i, m_row, v_row,
                                length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
            Rii = Rinv[i, i] * (si * si)
            tr += Rii * Jii
            quad += Jii * yi * yi

            for j in range(i):
                Jij = _Jij_matern25(w1, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                sj = Izi[j]
                Rij = Rinv[i, j] * (si * sj)
                yj = y_scaled[j]
                tr += 2.0 * Rij * Jij
                quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new