from numba import njit, prange, config, set_num_threads
import numpy as np
from math import erf, sqrt, pi, exp
from numpy.random import randn
from scipy.linalg import pinvh, cholesky
from .vecchia import K_matrix_nb, K_vec_nb
import itertools
from psutil import cpu_count

core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num, max_threads)
config.THREADING_LAYER = 'tbb'
set_num_threads(core_num)

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
def ghdiag(fct,mu,var,y):
    x, w = np.polynomial.hermite.hermgauss(10)
    N = np.shape(mu)[1]
    const = np.pi**(-0.5*N)
    xn = np.array(list(itertools.product(*(x,)*N)))
    wn = np.prod(np.array(list(itertools.product(*(w,)*N))), 1)[:, None]
    fn = sqrt(2.0)*(np.sqrt(var[:,None])*xn) + mu[:,None]
    llik=fct(y[:,None],fn)
    return np.sum(np.exp(np.log((wn * const)[None,:]) + llik), axis=1)

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
    Rinv=pinvh(R,check_finite=False)
    sigma2 = (1/np.diag(Rinv)).reshape(-1,1)
    sigma2 = scale*sigma2
    return sigma2

######functions for predictions########
@njit(cache=True,fastmath=True)
def k_one_vec(X,z,length,name):
    """Compute cross-correlation matrix between the testing and training input data.
    """
    if name=='sexp':
        X_l=X/length
        z_l=z/length
        L_X=np.expand_dims(np.sum(X_l**2,axis=1),axis=1)
        L_z=np.sum(z_l**2,axis=1)
        dis2=L_X-2*np.dot(X_l,z_l.T)+L_z
        k=np.exp(-dis2)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        m=len(z)
        X_l=np.expand_dims((X/length).T,axis=2)
        z_l=np.expand_dims((z/length).T,axis=2)
        k1=np.ones((n,m))
        k2=np.zeros((n,m))
        for i in range(d):
            dis=np.abs(X_l[i]-z_l[i].T)
            k1*=(1+sqrt(5)*dis+5/3*dis**2)
            k2+=dis
        k2=np.exp(-sqrt(5)*k2)
        k=k1*k2
    return k

def cond_mean(x,z,w1,global_w1,Rinv_y,length,name):
    """Make GP predictions.
    """
    if z is not None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    r=k_one_vec(w1,x,length,name)
    m=np.dot(Rinv_y, r)
    return m

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
    m, v = np.empty(n_pred), np.empty(n_pred)
    for i in prange(n_pred):
        ri=K_vec_nb(w1,x[i],name)
        Rinv_ri=np.dot(Rinv,ri)
        r_Rinv_r=np.dot(ri, Rinv_ri)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = abs(scale*(1.0+nugget-r_Rinv_r))
    return m, v

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
        denI = np.empty(d, dtype=np.float64)     # thread-local
        denJ = np.empty(d, dtype=np.float64)     # thread-local

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

SQRT5 = 2.2360679774997898
FIVE_THIRDS = 1.6666666666666667  # 5/3
INV_SQRT2 = 0.7071067811865475
INV_SQRT2PI = 0.3989422804014327  # 1/sqrt(2*pi)

@njit(cache=True, fastmath=True, inline="always")
def _Phi(x):
    """Standard normal CDF via erf. (SAME formula)"""
    return 0.5 * (1.0 + erf(x * INV_SQRT2))


@njit(cache=True, fastmath=True, inline="always")
def _matern25_det_1d(absdiff_scaled):
    """
    Deterministic matern2.5 kernel in 1D with r = |x-mu|/ell already scaled:
      (1 + sqrt(5) r + 5 r^2/3) * exp(-sqrt(5) r)
    (SAME as earlier)
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

    # polynomial pieces (SAME structure)
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

    # ===== common scalars (CHANGED: hoist repeats) =====
    den = 1.0 / (9.0 * ell4)                      # CHANGED
    sigma = sqrt(z_v)                        # CHANGED
    inv_sigma = 1.0 / sigma                       # CHANGED
    inv_sqrt_2zv = 1.0 / sqrt(2.0 * z_v)     # CHANGED
    pdf_scale = sqrt(0.5 * z_v / pi)    # CHANGED

    # CHANGED: reuse sums/products/powers
    S = x1 + x2                                   # CHANGED
    D = x2 - x1                                   # CHANGED
    P = x1 * x2                                   # CHANGED
    x1_sq = x1 * x1                               # CHANGED
    x2_sq = x2 * x2                               # CHANGED
    x2_cu = x2_sq * x2                            # CHANGED

    # CHANGED: shared exponent pieces
    base10 = 10.0 * z_v * inv_ell2                # CHANGED
    shiftS = SQRT5 * (S - 2.0 * z_m) * inv_ell    # CHANGED

    # ---- E3 block (SAME algebra; CHANGED: use den, reuse S,P,x1_sq,x2_sq) ----
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

    muC = z_m - 2.0 * SQRT5 * z_v * inv_ell  # SAME

    # CHANGED: avoid ** for muC powers
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

    # CHANGED: exp(base10 ± shiftS) reuse
    exp1 = exp(base10 + shiftS)  # SAME value
    uC = (muC - x2) * inv_sqrt_2zv     # CHANGED: reuse inv_sqrt_2zv
    pdfC = pdf_scale * exp(-0.5 * ((x2 - muC) * inv_sigma) ** 2)  # CHANGED: standardised

    P1 = exp1 * (
        0.5 * E3A31 * (1.0 + erf(uC)) +
        E3A32 * pdfC
    )

    # ---- E4 block (SAME; CHANGED: reuse den,S,D,P and x*_sq) ----
    E40 = 1.0 + (25.0 * x1_sq * x2_sq
                 + 3.0 * SQRT5 * (3.0 * ell3 - 5.0 * ell * P) * D
                 + 15.0 * ell2 * (x1_sq + x2_sq - 3.0 * P)) * den

    E41 = 5.0 * (3.0 * SQRT5 * ell * (x2_sq - x1_sq)
                 + 3.0 * ell2 * S
                 - 10.0 * P * S) * den

    E42 = 5.0 * (5.0 * x1_sq + 5.0 * x2_sq - 3.0 * ell2
                 - 3.0 * SQRT5 * ell * D
                 + 20.0 * P) * den

    E43 = -50.0 * S * den                 # CHANGED: X1+X2 == x1+x2 == S
    E44 = 25.0 * den

    # CHANGED: avoid ** for z_m powers
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

    exp2 = exp(-SQRT5 * D * inv_ell)  # SAME value

    u2 = (x2 - z_m) * inv_sqrt_2zv         # CHANGED
    u1 = (x1 - z_m) * inv_sqrt_2zv         # CHANGED

    pdf1 = pdf_scale * exp(-0.5 * ((x1 - z_m) * inv_sigma) ** 2)  # CHANGED
    pdf2 = pdf_scale * exp(-0.5 * ((x2 - z_m) * inv_sigma) ** 2)  # CHANGED

    P2 = exp2 * (
        0.5 * E4A41 * (erf(u2) - erf(u1)) +
        E4A42 * pdf1 -
        E4A43 * pdf2
    )

    # ---- E5 block (SAME; CHANGED: reuse den,S,P and avoid **) ----
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

    muD = z_m + 2.0 * SQRT5 * z_v * inv_ell  # SAME

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

    exp3 = exp(base10 - shiftS)  # CHANGED: reuse base10/shiftS

    uD = (x1 - muD) * inv_sqrt_2zv
    pdfD = pdf_scale * exp(-0.5 * ((x1 - muD) * inv_sigma) ** 2)

    P3 = exp3 * (
        0.5 * E5A51 * (1.0 + erf(uD)) +
        E5A52 * pdfD
    )

    return P1 + P2 + P3


@njit(cache=True, fastmath=True)
def Jd0_nb(x1, z_m, z_v, ell, ell2, ell3, ell4, inv_ell, inv_ell2):
    """Diagonal J component in 1D for Matern2.5. (SAME math)"""

    den = 1.0 / (9.0 * ell4)                      # CHANGED
    sigma = sqrt(z_v)                        # CHANGED
    inv_sigma = 1.0 / sigma                       # CHANGED
    inv_sqrt_2zv = 1.0 / sqrt(2.0 * z_v)     # CHANGED
    pdf_scale = sqrt(0.5 * z_v / pi)    # CHANGED

    x1_sq = x1 * x1                               # CHANGED
    x1_4 = x1_sq * x1_sq                          # CHANGED

    base10 = 10.0 * z_v * inv_ell2                # CHANGED
    shift2 = SQRT5 * (2.0 * x1 - 2.0 * z_m) * inv_ell  # CHANGED

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
                       inv_len_w[k], inv_len2_w[k])  # CHANGED args
    return out


@njit(cache=True, fastmath=True, inline="always")
def _Jij_matern25(w1, i, j, m_row, v_row,
                  length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w):
    """
    Compute J_ij (i!=j) without materialising J.
    CHANGED: uses precomputed length terms; still uses your Jd name (now with extra args).
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
                      inv_len_w[k], inv_len2_w[k])  # CHANGED args
    return out


# ============================================================
# matern2.5 fast: z is None (4 functions style like sexp)
#   CHANGED: IR fused into the i-loop that also does quad/tr
# ============================================================

@njit(cache=True, fastmath=True)
def link_gp_matern25_noz_serial(m, v, w1, Rinv, Rinv_y,
                               scale, length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w,
                               nugget):
    """
    Matérn-2.5 linked GP without forming I/J; z is None.
    CHANGED: single outer loop over i does IR + (diag/offdiag) quad/tr.
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

    for t in range(n_pred):
        m_row = m[t]
        v_row = v[t]

        IR = 0.0
        quad = 0.0
        tr = 0.0

        for i in range(n):
            # ---- IR part (CHANGED: fused into this loop) ----
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
def link_gp_matern25_noz_parallel(m, v, w1, Rinv, Rinv_y,
                                 scale, length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w,
                                 nugget):
    """Same as serial but prange over prediction points."""
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

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
#   CHANGED: same scaling trick as sexp (y_scaled and Rij scaled by si*sj)
#   CHANGED: IR fused into i-loop as well
# ============================================================

@njit(cache=True, fastmath=True)
def link_gp_matern25_withz_serial(m, v, z, w1, global_w1, Rinv, Rinv_y,
                                 scale,
                                 length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w,
                                 inv_len_z,
                                 nugget):
    """
    Matérn-2.5 linked GP with global z (deterministic Matérn for global part),
    without forming I/J or outer(Izi,Izi).
    """
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

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
            Rii = Rinv[i, i] * (si * si)          # CHANGED: scale Rij by si*sj
            tr += Rii * Jii
            quad += Jii * yi * yi

            # off-diagonal
            for j in range(i):
                Jij = _Jij_matern25(w1, i, j, m_row, v_row,
                                    length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w)
                sj = Izi[j]
                Rij = Rinv[i, j] * (si * sj)      # CHANGED: scale Rij by si*sj
                yj = y_scaled[j]
                tr += 2.0 * Rij * Jij
                quad += 2.0 * Jij * yi * yj

        m_new[t] = IR
        v_new[t] = abs(quad - IR * IR + scale * (1.0 + nugget - tr))

    return m_new, v_new


@njit(cache=True, parallel=True, fastmath=True)
def link_gp_matern25_withz_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y,
                                   scale,
                                   length_w, inv_len_w, inv_len2_w, len2_w, len3_w, len4_w,
                                   inv_len_z,
                                   nugget):
    n_pred, d = m.shape
    n = w1.shape[0]
    m_new = np.empty(n_pred, dtype=np.float64)
    v_new = np.empty(n_pred, dtype=np.float64)

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