from numba import njit, prange, config, set_num_threads
import numpy as np
from math import erf, sqrt, pi
from numpy.random import randn
from scipy.linalg import pinvh
from .vecchia import K_matrix_nb, quad, K_vec_nb, Jd, Jd0
import itertools
from psutil import cpu_count

core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num, max_threads)
config.THREADING_LAYER = 'workqueue'
set_num_threads(core_num)
#######functions for optim#########
@njit(cache=True)
def pdist_matern_coef(X):
    n = X.shape[0]
    out_size = np.int32((n * (n - 1)) / 2)
    dm = np.empty(out_size)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dm[k] = matern_coef(X[i], X[j])
            k += 1
    return dm

@njit(cache=True)
def matern_coef(v,u):
    dist = 1
    for r in range(len(v)):
        disi = np.abs(v[r] - u[r])
        dist *= 1+np.sqrt(5)*disi+(5/3)*disi**2
    return dist

@njit(cache=True)
def fod_exp(X,K):
    n, D = X.shape
    dm = np.zeros((D,n,n))
    for d in range(D):
        for i in range(n - 1):
            for j in range(i + 1, n):
                temp = 2*(X[i,d]-X[j,d])**2*K[i,j]
                dm[d,i,j], dm[d,j,i] = temp, temp
    return dm

@njit(cache=True)
def pdist_matern_one(X):
    n = X.shape[0]
    dm = np.zeros((n,n))
    dm1 = np.zeros((1,n,n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            temp1,temp2 = matern_one(X[i], X[j])
            dm[i,j], dm[j,i] = temp1, temp1 
            dm1[:,i,j], dm1[:,j,i] = temp2, temp2
    return dm, dm1

@njit(cache=True)
def pdist_matern_multi(X):
    n, D = X.shape
    dm = np.zeros((n,n))
    dm2 = np.zeros((D,n,n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            temp1,temp2=matern_multi(X[i], X[j])
            dm[i,j], dm[j,i] = temp1, temp1 
            dm2[:,i,j], dm2[:,j,i] = temp2, temp2
    return dm, dm2

@njit(cache=True)
def matern_one(v,u):
    dist = 1
    dist1 = 0
    for d in range(len(v)):
        disi = np.abs(v[d] - u[d])
        coefi = 1+np.sqrt(5)*disi+(5/3)*disi**2
        dist *= coefi
        coefi1=(5/3)*(disi**2)*(1+np.sqrt(5)*disi)/(1+np.sqrt(5)*disi+5/3*disi**2)
        dist1 += coefi1
    return dist, dist1

@njit(cache=True)
def matern_multi(v,u):
    dist = 1
    dist1=np.zeros(len(v)) 
    for d in range(len(v)):
        disi = np.abs(v[d] - u[d])
        coefi = 1+np.sqrt(5)*disi+(5/3)*disi**2
        dist *= coefi
        coefi1=(5/3)*(disi**2)*(1+np.sqrt(5)*disi)/(1+np.sqrt(5)*disi+5/3*disi**2)
        dist1[d] = coefi1
    return dist, dist1

@njit(cache=True)
def g(coef1, coef2, x, name):
    if name=='ga':
        return np.sum(coef1*np.log(x)-coef2*x)
    else:
        return np.sum(-coef1*np.log(x)-coef2/x)

######functions for imputer########
@njit(cache=True)
def fmvn_mu(mu,cov):
    """Generate multivariate Gaussian random samples with means.
    """
    d=len(cov)
    sn=randn(d,1)
    L=np.linalg.cholesky(cov)
    samp=(L@sn).flatten()+mu
    return samp

@njit(cache=True)
def fmvn(cov):
    """Generate multivariate Gaussian random samples without means.
    """
    d=len(cov)
    sn=randn(d,1)
    L=np.linalg.cholesky(cov)
    samp=(L@sn).flatten()
    return samp

######functions for classification########
@njit(cache=True)
def categorical_sampler(pvals):
    """
    Perform categorical sampling using numpy.random.multinomial inside Numba for efficiency.
    
    Parameters:
    pvals (numpy.ndarray): A 2D array of shape (num_samples, num_classes) representing the probability vectors for each sample.
    
    Returns:
    numpy.ndarray: A 1D array of sampled categorical outcomes (class indices).
    """
    num_samples = pvals.shape[0]
    samples = np.zeros(num_samples, dtype=np.int32)
    
    for i in range(num_samples):
        # Perform a multinomial sample with n=1 for each set of probabilities
        sample = np.random.multinomial(1, pvals[i])
        # Get the index of the selected category (one-hot encoded result)
        samples[i] = np.argmax(sample)
    
    return samples

@njit(cache=True)
def categorical_sampler_3d(pvals):
    """
    Perform categorical sampling on a 3D array of probability vectors where each 2D slice
    along the first dimension represents a sample.

    Parameters:
    pvals (numpy.ndarray): A 3D array of shape (num_samples, num_points, num_classes),
                           where each 2D slice represents the probability vectors for 
                           a sample, and each row in the 2D slice represents a point.

    Returns:
    numpy.ndarray: A 2D array of sampled categorical outcomes (class indices) with shape (num_points, num_samples),
                   where each row represents the sampled outcomes for each point across samples.
    """
    num_samples = pvals.shape[0]
    num_points = pvals.shape[1]
    samples = np.zeros((num_points, num_samples), dtype=np.int32)
    
    for i in range(num_samples):
        for j in range(num_points):
            # Perform a multinomial sample with n=1 for each set of probabilities
            sample = np.random.multinomial(1, pvals[i, j])
            # Get the index of the selected category (one-hot encoded result)
            samples[j, i] = np.argmax(sample)
    
    return samples

def logloss(probs, truth, ord = None):
    if ord is not None:
        probs = probs[:, ord, :]
    num_samples, num_points, _ = probs.shape
    true_class_probs = probs[np.arange(num_samples)[:, None], np.arange(num_points), truth]
    log_loss_values = -np.log(true_class_probs)
    average_log_loss = log_loss_values.mean()
    return average_log_loss

#@jit(nopython=True,cache=True)
#def fmvn_mu(mu,cov):
#    """Generate multivariate Gaussian random samples with means.
#    """
#    d=len(cov)
#    sn=randn(d,1)
#    U, s, _ = np.linalg.svd(cov)
#    samp=((U*np.sqrt(s))@sn).flatten() + mu
#    return samp

#@jit(nopython=True,cache=True)
#def fmvn(cov):
#    """Generate multivariate Gaussian random samples without means.
#    """
#    d=len(cov)
#    sn=randn(d,1)
#    U, s, _ = np.linalg.svd(cov)
#    samp=((U*np.sqrt(s))@sn).flatten()
#    return samp

@njit(cache=True)
def update_f(f,nu,theta):
    """Update ESS proposal samples.
    """
    fp=f*np.cos(theta) + nu*np.sin(theta)
    return fp

#@njit(cache=True)
#def Z_fct(X,W,b,length,M):
#    W=W/length
#    Z=np.dot(X,W.T)+b
 #   return np.sqrt(2/M)*np.cos(Z)

#@njit(cache=True)
#def cholesky_nb(X):
#    return np.linalg.cholesky(X)

@njit(cache=True)
def logdet_nb(L):
    return 2*np.sum(np.log(np.abs(np.diag(L))))

@njit(cache=True)
def trace_nb(K):
    n = K.shape[0]
    traces = np.empty(n, dtype=K.dtype)
    for i in range(n):
        traces[i] = np.trace(K[i])
    return traces

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
@njit(cache=True)
def Pmatrix(X):
    N,D=np.shape(X)
    P=np.empty((D,N,N))
    for d in range(D):
        for k in range(N):
            for l in range(k+1):
                temp = X[k,d]+X[l,d]
                if k==l:
                    P[d,k,l] = temp
                else:
                    P[d,k,l] = temp
                    P[d,l,k] = temp
    return P

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

#@jit(nopython=True,cache=True,fastmath=True)
#def gp(x,z,w1,global_w1,Rinv,Rinv_y,scale,length,nugget,name):
#    """Make GP predictions.
#    """
#    if z is not None:
#        x=np.concatenate((x, z),1)
#        w1=np.concatenate((w1, global_w1),1)
#    r=k_one_vec(w1,x,length,name)
#    Rinv_r=np.dot(Rinv,r)
#    r_Rinv_r=np.sum(r*Rinv_r,axis=0)
#    v=np.abs(scale*(1+nugget-r_Rinv_r))
#    #m=np.sum(r*Rinv_y,axis=0) 
 #   m=np.dot(Rinv_y, r)
#    return m, v

@njit(cache=True)
def gp_non_parallel(x,z,w1,global_w1,Rinv,Rinv_y,scale,length,nugget,name):
    """Make GP predictions
    """
    if z is not None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in range(n_pred):
        ri=K_vec_nb(w1,x[i],length,name)
        Rinv_ri=np.dot(Rinv,ri)
        r_Rinv_r=np.dot(ri, Rinv_ri)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = np.abs(scale*(1+nugget-r_Rinv_r))[0]
    return m, v

@njit(cache=True)
def link_gp_non_parallel(m, v, z, w1, global_w1, Rinv, Rinv_y, R2sexp, Psexp, scale, length, nugget, name):
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
        if z is not None:
            Izi = K_vec_nb(global_w1, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            if name == 'sexp' and R2sexp is not None and Psexp is not None:
                Ii,Ji = IJ_sexp(w1, m[i], v[i], length[:-Dz], R2sexp, Psexp)
            else:
                Ii,Ji = IJ_matern(w1, m[i], v[i], length[:-Dz])
            Ii *= Izi
            Ji *= Jzi
        else:
            if name == 'sexp' and R2sexp is not None and Psexp is not None:
                Ii,Ji = IJ_sexp(w1, m[i], v[i], length, R2sexp, Psexp)
            else:
                Ii,Ji = IJ_matern(w1, m[i], v[i], length)
        tr_RinvJ = trace_sum(Rinv,Ji)
        IRinv_y = np.dot(Ii,Rinv_y)
        m_new[i] = IRinv_y
        v_new[i] = np.abs(quad(Ji,Rinv_y)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new,v_new

@njit(cache=True, parallel=True)
def gp(x,z,w1,global_w1,Rinv,Rinv_y,scale,length,nugget,name):
    """Make GP predictions
    """
    if z is not None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in prange(n_pred):
        ri=K_vec_nb(w1,x[i],length,name)
        Rinv_ri=np.dot(Rinv,ri)
        r_Rinv_r=np.dot(ri, Rinv_ri)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = np.abs(scale*(1+nugget-r_Rinv_r))[0]
    return m, v

@njit(cache=True, parallel=True)
def link_gp(m, v, z, w1, global_w1, Rinv, Rinv_y, R2sexp, Psexp, scale, length, nugget, name):
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
        if z is not None:
            Izi = K_vec_nb(global_w1, z[i], length[-Dz::], name)
            Jzi = np.outer(Izi,Izi)
            if name == 'sexp' and R2sexp is not None and Psexp is not None:
                Ii,Ji = IJ_sexp(w1, m[i], v[i], length[:-Dz], R2sexp, Psexp)
            else:
                Ii,Ji = IJ_matern(w1, m[i], v[i], length[:-Dz])
            Ii *= Izi
            Ji *= Jzi
        else:
            if name == 'sexp' and R2sexp is not None and Psexp is not None:
                Ii,Ji = IJ_sexp(w1, m[i], v[i], length, R2sexp, Psexp)
            else:
                Ii,Ji = IJ_matern(w1, m[i], v[i], length)
        tr_RinvJ = trace_sum(Rinv,Ji)
        IRinv_y = np.dot(Ii,Rinv_y)
        m_new[i] = IRinv_y
        v_new[i] = np.abs(quad(Ji,Rinv_y)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new,v_new

@njit(cache=True)
def IJ_sexp(X, z_m, z_v, length, R2sexp, Psexp):
    n, d = X.shape
    I = np.zeros(n)
    J = np.zeros((n,n))
    X_z = X-z_m
    I_coef1, J_coef1 = 1., 1.
    for k in range(d):
        div = 2*z_v[k]/length[k]**2
        I_coef1 *= 1 + div
        J_coef1 *= 1 + 2*div
        J += (Psexp[k]-2*z_m[k]/length[k])**2/(2+4*div)
    I_coef1, J_coef1 = 1/sqrt(I_coef1), 1/sqrt(J_coef1)
    J = J_coef1 * np.exp(-J) * R2sexp
    for i in range(n):
        I_coef2 = 0.
        for k in range(d):
            I_coef2 += X_z[i,k]**2/(2*z_v[k]+length[k]**2)
        I[i] = I_coef1 * np.exp(-I_coef2)
    return I,J

@njit(cache=True)
def IJ_matern(X, z_m, z_v, length):
    n, d = X.shape
    I = np.zeros(n)
    J = np.zeros((n,n))
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

@njit(cache=True,fastmath=True)
def trace_sum(A,B):
    n = len(A)
    a = 0
    for k in range(n):
        for l in range(k+1):
            if k==l:
                a += A[k,l]*B[k,l]
            else:
                a += 2*A[k,l]*B[k,l]
    return a

@njit(cache=True, parallel=True)
def esloo_calculation(mu_i, var_i, Y, indices, start_rows):
    B = len(mu_i)
    mu = np.sum(mu_i,axis=0)/B
    sigma2 = np.sum((np.square(mu_i)+var_i),axis=0)/B-mu**2
    n, d = mu.shape
    L = Y.shape[0]
    if indices is not None:
        nesloo = np.zeros((L, d))
    else:
        nesloo = np.zeros((n, d))
    final_nesloo = np.empty_like(nesloo)
    seq = np.arange(L)
    reorder_idx = np.arange(L)
    for i in prange(n):
        if indices is not None:
            idx = indices==i
            f = Y[idx,:]
            reorder_idx_i = seq[idx]
        else:
            f = Y[i:i+1,:]
        mu_ii, var_ii = mu_i[:,i:i+1,:], var_i[:,i:i+1,:]
        esloo = sigma2[i] + (mu[i] - f)**2 #2d array
        normaliser = np.zeros((B, f.shape[0], f.shape[1])) #3d array
        for j in range(B):
            for k in range(f.shape[0]):
                error_jk = (mu_ii[j] - f[k])**2
                normaliser[j,k] = error_jk**2 + 6 * error_jk * var_ii[j] + 3 * var_ii[j]**2
        normaliser = np.sum(normaliser, axis=0)/B - esloo**2
        nesloo_ii = esloo / np.sqrt(normaliser)
        count = nesloo_ii.shape[0]
        starting_row =  start_rows[i]
        nesloo[starting_row:starting_row+count, :] = nesloo_ii
        if indices is not None:
            reorder_idx[starting_row:starting_row+count] = reorder_idx_i
    final_nesloo[reorder_idx,:] = nesloo
    return final_nesloo

#@jit(nopython=True,cache=True)
#def I_sexp_parallel(z_v,length,X_zi):
#    Id=1
#    for d in range(len(length)):
#        Id*=1/np.sqrt(1+2*z_v[d]/length[d]**2)*np.exp(-X_zi[d]**2/(2*z_v[d]+length[d]**2))
#    return Id

#@jit(nopython=True,cache=True)
#def J_sexp_parallel(z_v,length,X_zi,X_zj):
#    Jd=1
#    for d in range(len(length)):
#        dis1, dis2 = (X_zi[d]+X_zj[d])**2, (X_zi[d]-X_zj[d])**2
#        Jd*=1/np.sqrt(1+4*z_v[d]/length[d]**2)*np.exp(-dis1/(2*length[d]**2+8*z_v[d])-dis2/(2*length[d]**2))
#    return Jd