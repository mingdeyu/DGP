from numba import njit, vectorize, float64, prange, config
import numpy as np
from math import erf, exp, sqrt, pi
from numpy.random import randn
from scipy.linalg import pinvh
import itertools
import jax 
import jax.numpy as jnp
from functools import partial

def jax_k_one_vec(X,z,length):
    """Compute cross-correlation matrix between the testing and training input data.
    """
    X_l=X/length
    z_l=z/length
    L_X=jnp.expand_dims(jnp.sum(X_l**2,axis=1),axis=1)
    L_z=jnp.sum(z_l**2,axis=1)
    dis2=L_X-2*jnp.dot(X_l,z_l.T)+L_z
    k=jnp.exp(-dis2)
    return k


def jax_gp_predict(x,w1,Rinv,Rinv_y,scale,length,nugget):
    """Make GP predictions.
    """
    r=jax_k_one_vec(w1,x,length)
    Rinv_r=jnp.dot(Rinv,r)
    r_Rinv_r=jnp.sum(r*Rinv_r,axis=0)
    v=jnp.abs(scale*(1+nugget-r_Rinv_r)) 
    m=jnp.dot(Rinv_y, r)
    return m, v


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



def jax_I_sexp(X,z_m,z_v,length):
    # X: w imputated random variable
    # z_m: mu from last layer GP
    # z_v: var from last layer GP
    v_l=1/(1+2*z_v/length**2)
    X_l=X/length
    m_l=z_m/length
    cross1=jnp.dot(X_l**2,v_l.T)
    cross2=2*jnp.dot(X_l,(m_l*v_l).T)
    L_z=jnp.sum(m_l**2*v_l,axis=1)
    coef = 0.5*jnp.sum(jnp.log(v_l),axis=1)
    dist=coef-cross1+cross2-L_z
    I=jnp.exp(dist.T)
    return I



def jax_J_sexp(X,z_m,z_v,length,Psexp,R2sexp):
    X=X.T
    d=len(X)
    vli=z_v/length**2
    mli=z_m/length
    J=R2sexp.copy()
    for i in range(d):
        J*=1/jnp.sqrt(1+4*vli[i])*jnp.exp(-(Psexp[i]-2*mli[i])**2/(2+8*vli[i]))
    return J


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


@jax.jit
def jax_link_gp_sexp(m,v,w1,Rinv,Rinv_y,R2sexp,Psexp,scale,length,nugget):
    """Make linked GP predictions for sexp kernels.
    z: external input
    m: mean of the last layer GP
    v: variance of the last layer GP
    w1: imputed random variables of the linked GP
    """
    M=len(m)
    # v_new=jnp.empty((M))
    Dw=np.shape(w1)[1]
    if len(length)==1:
        length=length*jnp.ones(Dw)
    I=jax_I_sexp(w1,m,v,length)
    IRinv_y=jnp.dot(I,Rinv_y)  
    # for i in range(M):
    #     mi=m[i]
    #     vi=v[i]
        # J=jax_J_sexp(w1,mi,vi,length,Psexp,R2sexp)
        # tr_RinvJ=trace_sum(Rinv,J)

        # H = jnp.abs(quad(J,Rinv_y)-IRinv_y[i]**2+scale*(1+nugget-tr_RinvJ))
        # v_new= v_new.at[i].set(H)
    return IRinv_y


def jax_lgp_pred_single(x, one_imputed_layer):
    num_layers = len(one_imputed_layer)
    assert num_layers == 2, "Only support 2 layers"

    num_gp_in_second_layer = len(one_imputed_layer[1])
    assert num_gp_in_second_layer == 1, "Only support 1 GP in the second layer"

    num_gp_in_first_layer = len(one_imputed_layer[0])
    mu_first_layer = jnp.zeros((1,num_gp_in_first_layer))
    var_first_layer = jnp.zeros((1,num_gp_in_first_layer))

    for p in range (num_gp_in_first_layer):
        Rinv = one_imputed_layer[0][p].Rinv
        Rinv_y = one_imputed_layer[0][p].Rinv_y
        scale = one_imputed_layer[0][p].scale
        length = one_imputed_layer[0][p].length
        w1 = one_imputed_layer[0][p].input
        nugget = one_imputed_layer[0][p].nugget
        m,v = jax_gp_predict(x, w1, Rinv,Rinv_y,scale,length, nugget)
        mu_first_layer = mu_first_layer.at[0,p].set(m[0])
        var_first_layer = var_first_layer.at[0,p].set(v[0])

    R2sexp = one_imputed_layer[1][0].R2sexp
    Psexp = one_imputed_layer[1][0].Psexp
    Rinv = one_imputed_layer[1][0].Rinv
    Rinv_y = one_imputed_layer[1][0].Rinv_y
    w1 = one_imputed_layer[1][0].input
    scale = one_imputed_layer[1][0].scale
    length = one_imputed_layer[1][0].length
    nugget = one_imputed_layer[1][0].nugget
    pred_mu = jax_link_gp_sexp(mu_first_layer, var_first_layer, 
                               w1, Rinv, Rinv_y, R2sexp, Psexp, scale, length, nugget)

    return pred_mu[0]

vmap_grad = jax.vmap(jax.grad(jax_lgp_pred_single, argnums=0), in_axes=(0, None))
def vmap_grad_single_imp(x, one_imputed_layer):
    assert x.ndim == 2, "x should be 2D"
    return jnp.squeeze(vmap_grad(x[:,None,:], one_imputed_layer))

# vmap_jax_lgp_pred_single = jax.vmap(jax_lgp_pred_single, in_axes=(0, None))

if __name__ == "__main__":
    from jax import random
    from jax.config import config
    config.update("jax_enable_x64", True)

    
    
    from dgpsi import dgp, kernel, combine, lgp, path, emulator, gp
    from mogp_emulator.ExperimentalDesign import LatinHypercubeDesign
    import numpy as np
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    # Define the benchmark function
    def test_f(x):
        x1, x2 = x[..., 0], x[..., 1]
        return jnp.sin(x1) * jnp.cos(x2) + jnp.sin(2 * x1) * jnp.cos(3 * x2)

    lhd = LatinHypercubeDesign([(-1,1),(-1,1)])

    x_train = lhd.sample(20)
    y_train = np.array(test_f(x_train))
    layer1=[kernel(length=np.array([1]),name='sexp', scale_est=True),
            kernel(length=np.array([1]),name='sexp', scale_est=True)]
    layer2=[kernel(length=np.array([1]),name='sexp', scale_est=True)]
    all_layer=combine(layer1,layer2)
    m=dgp(x_train,y_train[:,None],all_layer)
    m.train(N=150)
    final_layer_obj=m.estimate()
    emu=emulator(final_layer_obj,N=10)

    grid_eval_grid = np.linspace(-1, 1, 30)
    grid_eval_grid = np.meshgrid(grid_eval_grid, grid_eval_grid)
    grid_eval_grid = np.stack([grid_eval_grid[0].flatten(), grid_eval_grid[1].flatten()], axis=-1)

    grid_eval_grad = np.linspace(-1, 1, 10)
    grid_eval_grad = np.meshgrid(grid_eval_grad, grid_eval_grad)
    grid_eval_grad = np.stack([grid_eval_grad[0].flatten(), grid_eval_grad[1].flatten()], axis=-1)

    pred_mu = np.zeros(900)
    grad_pred = np.zeros((100,2))
    for i in range(900):
        pred_mu[i] = jax_lgp_pred_single(grid_eval_grid[i][None,:], emu.all_layer_set[0])
    

    grad_pred = vmap_grad_single_imp(grid_eval_grad, emu.all_layer_set[0])
   
    pred_mu = pred_mu.reshape(30,30)
    plt.contourf(grid_eval_grid[:,0].reshape(30,30), grid_eval_grid[:,1].reshape(30,30), pred_mu)
    plt.quiver(grid_eval_grad[:,0].reshape(10,10), grid_eval_grad[:,1].reshape(10,10), 
               grad_pred[:,0].reshape(10,10), grad_pred[:,1].reshape(10,10))
    plt.show()

