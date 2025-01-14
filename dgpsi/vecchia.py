from numba import njit, prange, config, vectorize, float64, set_num_threads
import numpy as np
from numpy.random import randn
from math import erf, sqrt, pi, exp
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
config.THREADING_LAYER = 'workqueue'
set_num_threads(core_num)

def get_pred_nn(query, x, m = 50, method = 'exact', size = 40, efSearch = 100, n_jobs = -1):
    n, d = x.shape
    m = min(m, n)
    if m==n:
        k = query.shape[0]
        NN = np.arange(m) + np.arange(k)[:, np.newaxis]
        NN %= m
    else:
        if FAISS_AVAILABLE:
            if method == 'exact':
                neigh = faiss.IndexFlatL2(d)
            elif method == 'approx':
                neigh = faiss.IndexHNSWFlat(d, size)
                neigh.hnsw.efSearch = efSearch
            neigh.add(x)
            _, NN = neigh.search(query, k=int(m))
        else:
            neigh = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)
            neigh.fit(x)
            NN = neigh.kneighbors(query, n_neighbors=m, return_distance=False)
    return NN

@njit(cache=True)
def nn_brute(x, m):
    n = x.shape[0]
    m = min(m, n-1)
    NNarray = np.full((n, m+1), -1)
    for i in range(n):
        dist = np.sum((x[:(i+1),:]-x[i,:])**2, axis=1)
        order = np.argsort(dist)
        NNarray[i,:min(m+1, i+1)] = order[:min(m+1, i+1)]
    return NNarray

@njit(cache=True)
def extract_NN_m(NN_mask, less_than_k_mask, m):
    n = NN_mask.shape[0]
    NN_m = np.empty((n,m))
    for i in range(n):
        NN_m[i,] = NN_mask[i][less_than_k_mask[i]][:m]
    return NN_m

def nn(x, m, method = 'exact', size = 40, efSearch = 100, n_jobs = -1):
    n, d = x.shape
    m, mult = min(m, n-1), 2

    NNarray = np.full((n, m + 1), -1, dtype=int)

    maxval = min(mult * m + 1, n)
    NNarray[:maxval] = nn_brute(x[:maxval], m)

    query_inds, msearch = np.arange(maxval, n), m

    if FAISS_AVAILABLE:
        while len(query_inds) > 0:
            max_query_inds = np.max(query_inds) + 1
            msearch = min(max_query_inds, 2*msearch)
            data_inds = np.arange(min(max_query_inds, n))
            if method == 'exact':
                neigh = faiss.IndexFlatL2(d)
            elif method == 'approx':
                neigh = faiss.IndexHNSWFlat(d, size)
                neigh.hnsw.efSearch = efSearch
            neigh.add(x[data_inds,:])
            _, NN = neigh.search(x[query_inds,:], k=int(msearch))
            less_than_k = NN <= query_inds[:, np.newaxis]
            less_than_k_valid = NN >= 0
            less_than_k = np.logical_and(less_than_k, less_than_k_valid)
            sum_less_than_k = np.sum(less_than_k, axis=1)
            ind_less_than_k = sum_less_than_k >= m+1
            NN_mask, less_than_k_mask, query_inds_mask = NN[ind_less_than_k,:], less_than_k[ind_less_than_k,:], query_inds[ind_less_than_k]
            NN_m = extract_NN_m(NN_mask, less_than_k_mask, m+1)
            NNarray[query_inds_mask,:] = NN_m
            query_inds = query_inds[~ind_less_than_k]
    else:
        neigh = NearestNeighbors(algorithm='kd_tree', n_jobs=n_jobs)
        while len(query_inds) > 0:
            max_query_inds = np.max(query_inds) + 1
            msearch = min(max_query_inds, 2*msearch)
            data_inds = np.arange(min(max_query_inds, n))
            neigh.fit(x[data_inds,:])
            NN = neigh.kneighbors(x[query_inds,:], n_neighbors=msearch, return_distance=False)
            less_than_k = NN <= query_inds[:, np.newaxis]
            sum_less_than_k = np.sum(less_than_k, axis=1)
            ind_less_than_k = sum_less_than_k >= m+1
            NN_mask, less_than_k_mask, query_inds_mask = NN[ind_less_than_k,:], less_than_k[ind_less_than_k,:], query_inds[ind_less_than_k]
            NN_m = extract_NN_m(NN_mask, less_than_k_mask, m+1)
            NNarray[query_inds_mask,:] = NN_m
            query_inds = query_inds[~ind_less_than_k]
    NNarray = np.fliplr(np.sort(NNarray))
    return(NNarray)

@njit(cache=True)
def forward_solve_sp(L, NNarray, b):
    n, m = L.shape
    x = np.zeros(n)
    for i in range(n):
        sumj = 0.0
        for j in range(1, min(i+1, m)):
            sumj += L[i, j] * x[NNarray[i, j]]
        x[i] = (b[i] - sumj) / L[i,0]
    return x

#@njit(cache=True)
def fmvn_mu_sp(X, NNarray, scale, length, nugget, name, mu):
    """Generate multivariate Gaussian random samples with means.
    """
    d = X.shape[0]
    sn = randn(d)
    L = L_matrix(X, NNarray, length, nugget, name)/np.sqrt(scale)
    samp = forward_solve_sp(L, NNarray, sn) + mu
    return samp

#@njit(cache=True)
def fmvn_sp(X, NNarray, scale, length, nugget, name):
    """Generate multivariate Gaussian random samples without means.
    """
    d = X.shape[0]
    sn = randn(d)
    L = L_matrix(X, NNarray, length, nugget, name)/np.sqrt(scale)
    samp = forward_solve_sp(L, NNarray, sn)
    return samp

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
def vecchia_llik(X, y, NNarray, scale, length, nugget, name):
    n = X.shape[0]
    quad, logdet = np.array([0.]), np.array([0.])
    for i in prange(n):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        xi, yi= X[idx,:], y[idx,:]
        Ki = K_matrix_nb(xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)  
        Liyi = forward_solve(Li, yi)
        quad += Liyi[-1]**2
        logdet += 2*np.log(np.abs(Li[-1,-1]))
    llik = -0.5*(logdet + quad/scale) 
    return llik

@njit(cache=True, parallel=True, fastmath=True)
def vecchia_nllik(X, y, NNarray, scale, length, nugget, name, scale_est, nugget_est):
    n = X.shape[0]
    p = len(length)
    if nugget_est:
        p += 1
    #quad, logdet = np.array([0.]), np.array([0.])
    dquad, dlogdet = np.zeros(p), np.zeros(p)

    idx = NNarray[0]
    idx = idx[idx>=0][::-1]
    xi, yi = X[idx,:], y[idx,:]
    Ki, dKi = dK_matrix_nb(xi, length, nugget, name, nugget_est)
    #dquadi, dlogdeti = np.zeros(p), np.zeros(p)
    for k in range(p):
        dquad[k], dlogdet[k] = ((yi/Ki)**2*dKi[k])[0,0], (dKi[k]/Ki)[0,0]
    quad = yi[0]**2/Ki[0]
    logdet = np.log(Ki[0])
    #dquad += dquadi
    #dlogdet += dlogdeti

    for i in prange(1,n):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        bsize = len(idx)
        xi, yi = X[idx,:], y[idx,:]
        Ki, dKi = dK_matrix_nb(xi, length, nugget, name, nugget_est)
        dquadi, dlogdeti = np.zeros(p), np.zeros(p)
        Li = np.linalg.cholesky(Ki)  # Perform Cholesky decomposition
        Liyi = forward_solve(Li, yi)
        Ii = np.zeros(bsize)
        Ii[-1] = 1.0
        LiIi = backward_solve(Li.T, Ii)
        for k in range(p):
            LidKi = forward_solve(Li, dKi[k]@LiIi)
            si = (Liyi.T@LidKi)[0]
            dquadi[k], dlogdeti[k] = (2*si*Liyi[-1]-LidKi[-1]*Liyi[-1]**2)[0], (LidKi[-1])[0]
        quad += Liyi[-1]**2
        logdet += 2*np.log(np.abs(Li[-1,-1]))
        dquad += dquadi
        dlogdet += dlogdeti
    if scale_est:
        scale = quad[0]/n
        nllik = 0.5*(logdet + n*np.log(scale))
        ndllik = 0.5*(dlogdet - dquad/scale)
    else:
        nllik = 0.5*(logdet + quad/scale) 
        ndllik = 0.5*(dlogdet - dquad/scale)
    return nllik, ndllik, np.array([scale])

@njit(cache=True,fastmath=True)
def K_vec_nb(X, z, length, name):
    """Compute cross-correlation matrix between the testing and training input data.
    """
    n1, d = X.shape
    X_l, z_l = X/length, z/length
    K_vec = np.zeros(n1)
    if name == 'sexp':
        for i in range(n1):
            dist = 0
            for k in range(d):
                dist += (X_l[i,k] - z_l[k])**2
            K_vec[i] = np.exp(-dist)
    elif name=='matern2.5':
        for i in range(n1):
            coef1, coef2 = 1, 0
            for k in range(d):
                distk = np.abs(X_l[i,k] - z_l[k])
                coef1 *= 1+np.sqrt(5)*distk+(5/3)*distk**2
                coef2 += distk
            K_vec[i] = coef1 * np.exp(-np.sqrt(5) * coef2)
    return K_vec

@njit(cache=True)
def K_cross_nb(x1, x2, length, name):
    n1, d = x1.shape
    n2, _ = x2.shape
    x1_l = x1/length
    x2_l = x2/length
    K = np.zeros((n1, n2))
    if name == 'sexp':
        for i in range(n1):
            for j in range(n2):
                dist = 0
                for k in range(d):
                    dist += (x1_l[i,k] - x2_l[j,k])**2
                K[i,j] = np.exp(-dist)
    elif name == 'matern2.5':
        for i in range(n1):
            for j in range(n2):
                coef1, coef2 = 1, 0
                for k in range(d):
                    distk = np.abs(x1_l[i,k] - x2_l[j,k])
                    coef1 *= 1+np.sqrt(5)*distk+(5/3)*distk**2
                    coef2 += distk
                K[i,j] = coef1 * np.exp(-np.sqrt(5) * coef2)
    return K

@njit(cache=True)
def K_matrix_nb(xi, length, nugget, name):
    n, d = xi.shape
    xi_l=xi/length
    K = np.zeros((n,n))
    if name == 'sexp':
        for i in range(n):
            for j in range(i + 1):
                if i==j:
                    K[i,j] = 1 + nugget
                else:
                    dist = 0
                    for k in range(d):
                        dist += (xi_l[i,k] - xi_l[j,k])**2
                    K[i,j] = np.exp(-dist)
                    K[j,i] = K[i,j]
    elif name == 'matern2.5':
        for i in range(n):
            for j in range(i + 1):
                if i==j:
                    K[i,j] = 1 + nugget
                else:
                    coef1, coef2 = 1, 0
                    for k in range(d):
                        distk = np.abs(xi_l[i,k] - xi_l[j,k])
                        coef1 *= 1+np.sqrt(5)*distk+(5/3)*distk**2
                        coef2 += distk
                    K[i,j] = coef1 * np.exp(-np.sqrt(5) * coef2)
                    K[j,i] = K[i,j]
    return K

@njit(cache=True)
def dK_matrix_nb(xi, length, nugget, name, nugget_est):
    n, d = xi.shape
    xi_l=xi/length
    K = np.zeros((n,n))
    if len(length)==1:
        if nugget_est:
            dK = np.zeros((2,n,n))
            for i in range(n):
                dK[1,i,i] = nugget
        else:
            dK = np.zeros((1,n,n))
        if name == 'sexp':
            for i in range(n):
                for j in range(i + 1):
                    if i==j:
                        K[i,j] = 1 + nugget
                    else:
                        dist = 0
                        for k in range(d):
                            dist += (xi_l[i,k] - xi_l[j,k])**2
                        K[i,j] = np.exp(-dist)
                        dK[0,i,j] = 2*dist*K[i,j]
                        K[j,i], dK[0,j,i] = K[i,j], dK[0,i,j]
        elif name == 'matern2.5':
            for i in range(n):
                for j in range(i + 1):
                    if i==j:
                        K[i,j] = 1 + nugget
                    else:
                        coef1, coef2, coef3 = 1, 0, 0
                        for k in range(d):
                            distk = np.abs(xi_l[i,k] - xi_l[j,k])
                            el1, el2 = 1+np.sqrt(5)*distk, (5/3)*distk**2
                            coef = el1 + el2
                            coef1 *= coef
                            coef2 += distk
                            coef3 += el2*el1/coef
                        K[i,j] = coef1 * np.exp(-np.sqrt(5) * coef2)
                        dK[0,i,j] = coef3*K[i,j]
                        K[j,i], dK[0,j,i] = K[i,j], dK[0,i,j]
    else:
        if nugget_est:
            dK = np.zeros((d+1,n,n))
            for i in range(n):
                dK[d,i,i] = nugget
        else:
            dK = np.zeros((d,n,n))
        if name == 'sexp':
            for i in range(n):
                for j in range(i + 1):
                    if i==j:
                        K[i,j] = 1 + nugget
                    else:
                        dist = 0
                        coef = np.zeros(d)
                        for k in range(d):
                            el = (xi_l[i,k] - xi_l[j,k])**2
                            dist += el
                            coef[k] = 2*el
                        K[i,j] = np.exp(-dist)
                        K[j,i] = K[i,j]
                        for k in range(d):
                            dK[k,i,j] = coef[k]*K[i,j]
                            dK[k,j,i] = dK[k,i,j]
        elif name == 'matern2.5':
            for i in range(n):
                for j in range(i + 1):
                    if i==j:
                        K[i,j] = 1 + nugget
                    else:
                        coef1, coef2, coef3 = 1, 0, np.zeros(d)
                        for k in range(d):
                            distk = np.abs(xi_l[i,k] - xi_l[j,k])
                            el1, el2 = 1+np.sqrt(5)*distk, (5/3)*distk**2
                            coef = el1 + el2
                            coef1 *= coef
                            coef2 += distk
                            coef3[k] = el2*el1/coef
                        K[i,j] = coef1 * np.exp(-np.sqrt(5) * coef2)
                        K[j,i] = K[i,j]
                        for k in range(d):
                            dK[k,i,j] = coef3[k]*K[i,j]
                            dK[k,j,i] = dK[k,i,j]
    return K, dK
    
@njit(cache=True, parallel=True)
def L_matrix(X, NNarray, length, nugget, name):
    n, m = NNarray.shape
    L_matrix = np.zeros((n, m))
    for i in prange(n):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        bsize = len(idx)
        Ii = np.zeros(bsize)
        Ii[-1] = 1.0
        xi = X[idx,:]
        Ki = K_matrix_nb(xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)
        LiIi = backward_solve(Li.T, Ii)
        L_matrix[i, :bsize] = LiIi.T[0][::-1]
    return L_matrix

@njit(cache=True, parallel=True)
def U_matrix(X, revNNarray, revCond, length, nugget, scale, gamma, name):
    n, m = revNNarray.shape
    U_matrix = np.zeros((n, m))
    for i in prange(n):
        idx = revNNarray[i]
        cond = revCond[i]
        idx = idx[idx>=0]
        cond = cond[idx>=0]
        bsize = len(idx)
        Ii = np.zeros(bsize)
        Ii[-1] = 1.0
        xi = X[idx,:]
        gammai = gamma[idx] * ~cond
        Ki = scale * K_matrix_nb(xi, length, nugget, name) + np.diag(gammai)
        Li = np.linalg.cholesky(Ki)
        LiIi = backward_solve(Li.T, Ii)
        U_matrix[i, :bsize] = LiIi.T[0]
    return U_matrix

#@njit(cache=True)
#def pointers(NNarray):
#    n, m = NNarray.shape
#    length = int((n-m)*m+m*(m+1)/2)
#    rows = np.zeros(length, dtype=np.int32)
#    cols = np.zeros(length, dtype=np.int32)
#    idx= 0
#    for i in range(n):
#        for j in range(min(i+1, m)):
#            rows[idx]=i
#            cols[idx]=NNarray[i, j]
#            idx += 1
#    return rows, cols

@njit(cache=True)
def imp_pointers(NNarray):
    n = NNarray.shape[0]
    revNNarray = NNarray[:,::-1]
    nentries = revNNarray.size
    rowpointers, colindices = np.zeros(nentries), np.zeros(nentries)
    cur = 0
    for i in range(n):
        idx = revNNarray[i]
        exist_n = idx>=0
        idx = idx[exist_n]
        n0 = len(idx)
        rowpointers[cur:cur+n0] = i
        colindices[cur:cur+n0] = idx
        cur = cur + n0
    return rowpointers, colindices

@njit(cache=True)
def imp_pointers_rep(NNarray, max_rep, rep, ord):
    n = NNarray.shape[0]
    rep_len = rep.shape[0]
    Cond = NNarray > n-1
    revNNarray = NNarray[:,::-1]
    revCond = Cond[:,::-1]
    nentries = revNNarray.size * max_rep
    rowpointers, colindices = np.full(nentries, -1), np.full(nentries, -1)
    cur = 0
    for i in range(n):
        idx = revNNarray[i]
        exist_n = idx>=0
        idx = idx[exist_n]
        revCond0 = revCond[i, exist_n]
        idx_latent = idx[revCond0]
        idx_obs = idx[~revCond0]
        selected_order_indices = ord[idx_obs]
        selected_order_set = set(selected_order_indices)
        mask = np.zeros(rep_len, dtype=np.bool_)
        for k in range(rep_len):
            if rep[k] in selected_order_set:
                mask[k] = True
        obs_positions = np.where(mask)[0]
        latent_position = idx_latent - n + rep_len
        n0 = len(obs_positions) + len(latent_position)
        cur_cols = np.empty(n0, dtype=np.int32)
        cur_cols[:len(obs_positions)] = obs_positions
        cur_cols[len(obs_positions):] = latent_position
        rowpointers[cur:cur+n0] = i
        colindices[cur:cur+n0] = cur_cols
        cur = cur + n0
    valid = rowpointers != -1
    rowpointers = rowpointers[valid]
    colindices = colindices[valid]

    return rowpointers, colindices

#def L_matrix_sp(X, NNarray, scale, length, nugget, name, rows, cols):
#    n = X.shape[0]
#    L = L_matrix(X, NNarray, length, nugget, name)/np.sqrt(scale)
#    data = L[NNarray!=-1]
#    L = csr_matrix((data, (rows, cols)), shape=(n, n))
#    return L

@njit(cache=True, parallel=True)
def U_matrix_rep(X, revNNarray, revCond, rep, ord, row, length, nugget, scale, gamma, name):
    n = revNNarray.shape[0]
    rep_len = rep.shape[0]
    U_matrix = np.zeros(len(row))
    for i in prange(n):
        idx = revNNarray[i]
        cond = revCond[i]
        idx = idx[idx>=0]
        cond = cond[idx>=0]
        idx_latent = idx[cond]
        idx_obs = idx[~cond]
        idx_latent_prev = idx_latent[:-1]
        idx_latent_current = idx_latent[-1]
        selected_order_indices = ord[idx_obs]
        selected_order_set = set(selected_order_indices)
        rep0 = np.full(rep_len, -1)
        for k in range(rep_len):
            if rep[k] in selected_order_set:
                rep0[k] = np.where(selected_order_indices == rep[k])[0][0]
        mask0 = rep0 != -1
        rep0_i = rep0[mask0]
        gamma_i = gamma[mask0]
        xi_obs = X[idx_obs,:]
        if len(idx_latent) == 1:
            x0 = X[idx_latent_current]
            Ki_obs = scale * K_matrix_nb(xi_obs, length, nugget, name)
            ki_obs_0 = scale * K_vec_nb(xi_obs, x0, length, name)
            Li_obs = np.linalg.cholesky(Ki_obs)
            Li_obs_mask = Li_obs[rep0_i,:]
            ki_obs_0_mask = ki_obs_0[rep0_i]
            ki_obs_0_mask_GammaInv = ki_obs_0_mask * (1/gamma_i)
            Li_obs_mask_GammaInv = Li_obs_mask.T * (1/gamma_i)
            Li_obs_mask_GammaInv_Li_obs_mask_Ii = Li_obs_mask_GammaInv@Li_obs_mask + np.eye(len(Li_obs))
            L_Li_obs_mask_GammaInv_Li_obs_mask_Ii = np.linalg.cholesky(Li_obs_mask_GammaInv_Li_obs_mask_Ii)
            Li_obs_mask_GammaInv_ki_obs_0_mask = np.dot(Li_obs_mask_GammaInv, ki_obs_0_mask)
            L_Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv_ki_obs_0_mask = backward_solve(L_Li_obs_mask_GammaInv_Li_obs_mask_Ii.T, forward_solve(L_Li_obs_mask_GammaInv_Li_obs_mask_Ii, Li_obs_mask_GammaInv_ki_obs_0_mask).flatten()).flatten()
            B = ki_obs_0_mask_GammaInv - np.dot(Li_obs_mask_GammaInv.T, L_Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv_ki_obs_0_mask)
            a = scale * (1 + nugget) - np.dot(B, ki_obs_0_mask)
            val = np.concatenate((-1/np.sqrt(a)*B, 1/np.sqrt(np.array([a]))))
        else:
            xi_latent = X[idx_latent_prev,:]
            x0 = X[idx_latent_current]
            Ki_obs = scale * K_matrix_nb(xi_obs, length, nugget, name)
            Ki_latent = scale * K_matrix_nb(xi_latent, length, nugget, name)
            ki_obs_latent = scale * K_cross_nb(xi_obs, xi_latent, length, name)
            ki_obs_0 = scale * K_vec_nb(xi_obs, x0, length, name)
            ki_latent_0 = scale * K_vec_nb(xi_latent, x0, length, name)
            Li_obs = np.linalg.cholesky(Ki_obs)
            Li_obs_mask = Li_obs[rep0_i,:]
            ki_obs_latent_mask = ki_obs_latent[rep0_i,:]
            ki_obs_0_mask = ki_obs_0[rep0_i]
            ki_obs_0_mask_GammaInv = ki_obs_0_mask * (1/gamma_i)
            Li_obs_mask_GammaInv = Li_obs_mask.T * (1/gamma_i)
            Li_obs_mask_GammaInv_Li_obs_mask_Ii = Li_obs_mask_GammaInv@Li_obs_mask + np.eye(len(Li_obs))
            Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv = np.linalg.solve(Li_obs_mask_GammaInv_Li_obs_mask_Ii, Li_obs_mask_GammaInv)
            Li_obs_mask_GammaInv_ki_obs_0_mask = np.dot(Li_obs_mask_GammaInv, ki_obs_0_mask)
            ki_obs_0_mask_A = ki_obs_0_mask_GammaInv - np.dot(Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv.T, Li_obs_mask_GammaInv_ki_obs_0_mask)
            ki_obs_latent_mask_GammaInv = ki_obs_latent_mask.T * (1/gamma_i)
            Li_obs_mask_Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv = np.dot(Li_obs_mask, Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv)
            ki_obs_latent_mask_A = ki_obs_latent_mask_GammaInv - np.dot(ki_obs_latent_mask_GammaInv, Li_obs_mask_Li_obs_mask_GammaInv_Li_obs_mask_Ii_Li_obs_mask_GammaInv)
            ki_obs_latent_mask_A_ki_obs_latent_mask = np.dot(ki_obs_latent_mask_A, ki_obs_latent_mask)
            Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask = Ki_latent - ki_obs_latent_mask_A_ki_obs_latent_mask
            L_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask = np.linalg.cholesky(Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask)
            ki_obs_latent_mask_ki_obs_0_mask_A = np.dot(ki_obs_latent_mask.T, ki_obs_0_mask_A)
            Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_obs_latent_mask_ki_obs_0_mask_A = backward_solve(L_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask.T, forward_solve(L_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask, ki_obs_latent_mask_ki_obs_0_mask_A).flatten()).flatten()
            ki_obs_latent_mask_A_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_obs_latent_mask_ki_obs_0_mask_A = np.dot(ki_obs_latent_mask_A.T, Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_obs_latent_mask_ki_obs_0_mask_A)
            Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_latent_0 = backward_solve(L_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask.T, forward_solve(L_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask, ki_latent_0).flatten()).flatten()
            ki_obs_latent_mask_A_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_latent_0 = np.dot(ki_obs_latent_mask_A.T, Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_latent_0)
            B1 = ki_obs_0_mask_A + ki_obs_latent_mask_A_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_obs_latent_mask_ki_obs_0_mask_A - ki_obs_latent_mask_A_Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_latent_0
            B2 = Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_latent_0 - Ki_latent_ki_obs_latent_mask_A_ki_obs_latent_mask_ki_obs_latent_mask_ki_obs_0_mask_A
            a = scale * (1 + nugget) - np.dot(B1, ki_obs_0_mask) - np.dot(B2, ki_latent_0)
            val = np.concatenate((-1/np.sqrt(a)*B1, -1/np.sqrt(a)*B2, 1/np.sqrt(np.array([a]))))
        U_matrix[row==i] = val
    return U_matrix

def U_matrix_sp_rep(X, NNarray, rep, ord, scale, length, nugget, name, gamma, rows, cols):
    n = X.shape[0]
    Cond = NNarray > n-1
    revNNarray = NNarray[:,::-1]
    revCond = Cond[:,::-1]
    data = U_matrix_rep(np.vstack((X, X)), revNNarray, revCond, rep, ord, rows, length, nugget, scale, gamma, name)
    U = csr_matrix((data, (cols, rows)), shape=(len(rep)+n, n))
    U_latent = U[-n:,:]
    U_obs_latent = U[:-n,:]
    return U_latent, U_obs_latent

def U_matrix_sp(X, NNarray, scale, length, nugget, name, gamma, rows, cols):
    n = X.shape[0]
    Cond = NNarray > n-1
    revNNarray = NNarray[:,::-1]
    revCond = Cond[:,::-1]
    U = U_matrix(np.vstack((X, X)), revNNarray, revCond, length, nugget, scale, gamma, name)
    data = U.flatten()
    U = csr_matrix((data, (cols, rows)), shape=(2*n, n))
    U_latent = U[n::,:]
    U_obs_latent = U[:n,:]
    return U_latent, U_obs_latent

def cond_mean_vecch(x, z, w1, global_w1, y, scale, length, nugget, name, m, nn_method):
    """Make GP mean predictions with Vecchia approximation in initialisation.
    """
    if z is not None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    NNarray = get_pred_nn(x/length, w1/length, m, method = nn_method)
    m,_ = gp_vecch(x, w1, NNarray, y, scale[0], length, nugget[0], name)
    return m

@njit(cache=True, parallel=True)
def gp_vecch(x,w,NNarray,y,scale,length,nugget,name):
    """Make GP predictions with Vecchia approximation.
    """
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in prange(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0]
        Xi = np.vstack((w[idx,:], x[i:i+1,:]))
        Ki = K_matrix_nb(Xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)
        yi = y[idx,0]
        m[i] = np.dot(Li[-1,:-1], forward_solve(Li[:-1, :-1], yi).flatten())
        v[i] = scale * Li[-1,-1]**2
    return m, v

@njit(cache=True, parallel=True)
def loo_gp_vecch(x,NNarray,y,scale,length,nugget,name):
    """Compute LOO for GP with Vecchia approximation.
    """
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in prange(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        Xi = x[idx,:]
        Ki = K_matrix_nb(Xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)
        yi = y[idx,0]
        m[i] = np.dot(Li[-1,:-1], forward_solve(Li[:-1, :-1], yi[:-1]).flatten())
        v[i] = scale * Li[-1,-1]**2
    return m, v

@njit(cache=True)
def gp_vecch_non_parallel(x,w,NNarray,y,scale,length,nugget,name):
    """Make GP predictions with Vecchia approximation.
    """
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in range(n_pred):
        idx = NNarray[i]
        idx = idx[idx>=0]
        Xi = np.vstack((w[idx,:], x[i:i+1,:]))
        Ki = K_matrix_nb(Xi, length, nugget, name)
        Li = np.linalg.cholesky(Ki)
        yi = y[idx,0]
        m[i] = np.dot(Li[-1,:-1], forward_solve(Li[:-1, :-1], yi).flatten())
        v[i] = scale * Li[-1,-1]**2
    return m, v

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
    x = np.zeros(n)  # Solution vector
    for i in range(n):
        sum_lx = 0.
        for j in range(L_indptr[i], L_indptr[i + 1]):
            if L_indices[j] < i:
                sum_lx += L_data[j] * x[L_indices[j]]
            elif L_indices[j] == i:
                x[i] = (b[i] - sum_lx) / L_data[j]
                break  # Only one diagonal element per row, so can break after processing it
    return x

@njit(cache=True)
def backward_substitute(U_data, U_indices, U_indptr, b):
    """
    Solves Ux = b for x, where U is an upper triangular matrix in CSR format.
    - U_data: Non-zero values of U.
    - U_indices: Column indices of non-zero values in U.
    - U_indptr: Index pointers for rows in U.
    - b: Right-hand side vector.
    Returns:
    - x: Solution vector.
    """
    n = len(U_indptr) - 1  # Number of rows
    x = np.zeros(n)  
    for i in range(n - 1, -1, -1):
        sum_lx = 0.0
        diag_val = None
        for j in range(U_indptr[i], U_indptr[i + 1]):
            if U_indices[j] > i:
                sum_lx += U_data[j] * x[U_indices[j]]
            elif U_indices[j] == i:
                diag_val = U_data[j]
            x[i] = (b[i] - sum_lx) / diag_val
    return x

#def rep_sp(indices):
#    num_rows = len(indices)
#    num_cols = np.max(indices) + 1

    # Create row and column indices for CSC format
#    rows = np.arange(num_rows)  # Each entry specifies the row in M
#    cols = indices  # Each index corresponds to a column in M

    # Data for the matrix
#    data = np.ones(num_rows)  # Place a '1' in each specified position

    # Create the CSC matrix
#    M_csc = csc_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
#    return M_csc
    
@njit(cache=True, parallel=True)
def link_gp_vecch(m, v, z, w1, global_w1, NNarray, y, scale, length, nugget, name):
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
        if z is not None:
            wi, global_wi = w1[idx,:], global_w1[idx,:]
            Izi = K_vec_nb(global_wi, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            Ii,Ji = IJ_nb(wi, m[i], v[i], length[:-Dz], name)
            Ii,Ji = Ii*Izi, Ji*Jzi
            Ki = K_matrix_nb(np.concatenate((wi, global_wi),1), length, nugget, name)
        else:
            wi = w1[idx,:]
            Ii,Ji = IJ_nb(wi, m[i], v[i], length, name)
            Ki = K_matrix_nb(wi, length, nugget, name)
        tr_RinvJ=np.trace(np.linalg.solve(Ki,Ji))
        Li = np.linalg.cholesky(Ki)
        Rinv_y = backward_solve(Li.T, forward_solve(Li, yi).flatten()).flatten()
        IRinv_y = np.dot(Ii,Rinv_y)
        m_new[i] = IRinv_y
        v_new[i] = np.abs(quad(Ji,Rinv_y)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new,v_new

@njit(cache=True)
def link_gp_vecch_non_parallel(m, v, z, w1, global_w1, NNarray, y, scale, length, nugget, name):
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
        if z is not None:
            wi, global_wi = w1[idx,:], global_w1[idx,:]
            Izi = K_vec_nb(global_wi, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            Ii,Ji = IJ_nb(wi, m[i], v[i], length[:-Dz], name)
            Ii,Ji = Ii*Izi, Ji*Jzi
            Ki = K_matrix_nb(np.concatenate((wi, global_wi),1), length, nugget, name)
        else:
            wi = w1[idx,:]
            Ii,Ji = IJ_nb(wi, m[i], v[i], length, name)
            Ki = K_matrix_nb(wi, length, nugget, name)
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