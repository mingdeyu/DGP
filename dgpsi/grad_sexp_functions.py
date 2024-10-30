from numba import njit, set_num_threads, config, prange, vectorize
import numpy as np
from psutil import cpu_count
from dgpsi.functions import k_one_vec, gp

core_num = cpu_count(logical = False)
config.THREADING_LAYER = 'workqueue'
set_num_threads(core_num)


@njit(cache=True)
def epsilon_sexp(x, z_m, z_v, length, scale, return_d_epsilon:bool=False):
    """ compute the epsilon function in the I function.
    N is the number of samples
    M is the number of prediction points
    D is the number of dimensions of the input
    P is the number of GP nodes in first layer

    input: x: the impuated random variable w (N, P) 
           z_m: the mu from the last layer GP (M, P) from x_star
           z_v: the var from the last layer GP (M, P) from x_star
           length: the length of the kernel (P,)
           return_d_epsilon: whether return the derivative of epsilon with respect to z_m and z_v
    return epsilon: the epsilon function in the I function (M, N, P)
           d_epsilon_dz_m: the derivative of epsilon with respect to z_m (M, N, P)
           d_epsilon_dz_v: the derivative of epsilon with respect to z_v (M, N, P)
    """
    N, P = x.shape
    M = z_m.shape[0]
    epsilon = np.zeros((M, N, P))
    if return_d_epsilon:
        d_epsilon_dz_m = np.zeros((M, N, P))
        d_epsilon_dz_v = np.zeros((M, N, P))
    for m in range(M):
        for n in range(N):
            for p in range(P):
                epsilon[m, n, p] = 1/(1+2*z_v[m, p]/length[p]**2) * np.exp(-(x[n, p]-z_m[m, p])**2/(2*z_v[m, p]+length[p]**2))
                if return_d_epsilon:
                    d_epsilon_dz_m[m,n,p] = 2*(z_m[m,p]-x[n,p])/(2*z_v[m,p]+length[p]**2) * epsilon[m,n,p]
                    d_epsilon_dz_v[m,n,p] = (1/(2*z_v[m,p]+length[p]**2)-2*(x[n,p]-z_m[m,p])**2/(2*z_v[m,p]+length[p]**2)**2) * epsilon[m,n,p]
    if return_d_epsilon:
        return epsilon*scale, d_epsilon_dz_m*scale, d_epsilon_dz_v*scale
    else:
        return epsilon*scale

# @njit(cache=True)
# def d_epsilon_dz_m(x, z_m, z_v, length):
#     """Compute the derivative of the epsilon function with respect to z_m.
#     input: x: the impuated random variable w (N, P) 
#            z_m: the mu from the last layer GP (M, P) from x_star
#            z_v: the var from the last layer GP (M, P) from x_star
#            length: the length of the kernel (P,)
#     return: d_epsilon_dz_m: the derivative of epsilon with respect to z_m (M, N, P)
#     """
#     N, P = x.shape
#     M = z_m.shape[0]
#     d_epsilon_dz_m = np.zeros((M, N, P))
#     for m in range(M):
#         for n in range(N):
#             for p in range(P):
#                 d_epsilon_dz_m[m, n, p] = (x[n, p]-z_m[m, p])/(z_v[m, p]+length[p]**2) * epsilon_sexp(x, z_m, z_v, length)[m, n, p]
#     return d_epsilon_dz_m

@njit(cache=True)
def sexp_k_one_vector_derivative(x_star, X, length, scale):
    """Compute the derivative of the squared exponential kernel.
    r(x) = [k(x, x_1), k(x, x_2), ..., k(x, x_n)]
    input: x_star: the input point (M, D)
    return ∇_x r(x) = [d/dx k(x, x_1), d/dx k(x, x_2), ..., d/dx k(x, x_n)] with shape (M, N, D)
    """
    # Ensure that x_star is a 2D array [M, D]
    assert x_star.ndim == 2, "x_star should be a 2D array with shape [M, D]"
    # Get the shapes
    M, D = x_star.shape
    N, _ = X.shape
    # Compute pairwise differences
    diff = x_star[:, None, :] - X[None, :, :]  # Shape: (M, N, D)
    # Compute squared exponential kernel values for all pairs
    sq_dist = np.sum(diff ** 2, axis=-1)  # Shape: (M, N)
    k_values = np.exp(-sq_dist / length**2)  # Shape: (M, N)
    # Compute the gradient
    grad_k = -2 * diff / length**2 * k_values[:, :, None] * scale # Broadcasting k_values to shape (M, N, D)
    return grad_k

@njit(cache=True)
def second_k_xx_derivatives(length, scale, output_shape):
    value = scale/length**2
    return value*np.ones(output_shape)



def nabla_sexp_I(x_star:np.array, all_layers:list):
    """Compute the nabla of the I function for the squared exponential kernel.
    input: x_star: the input point
           all_layers: all layers in the DGP model
    return: nabla_I: the nabla of the I function about x_star
    """
    assert len(all_layers) == 2, "Only support two layers now."
    assert len(all_layers[1]) == 1, "Only support one GP in the second layer now."
    first_layer = all_layers[0]
    second_layer = all_layers[1]
    
    global_input = first_layer[0].input
    M, _ = x_star.shape
    N, D = global_input.shape 
    nabla_I = np.zeros((M, N, D))

    # compute the output of the first layer given x_star
    num_gp_nodes = len(first_layer)
    mu_first_layer = np.zeros((M, num_gp_nodes))
    var_first_layer = np.zeros((M, num_gp_nodes))
    dmu_dx = np.zeros((M, num_gp_nodes, D, 1))
    dvar_dx = np.zeros((M, num_gp_nodes, D, D))
    lengths = np.zeros(num_gp_nodes)

    for p in range(num_gp_nodes):
        Rinv = first_layer[p].Rinv
        Rinv_y = first_layer[p].Rinv_y
        scale = first_layer[p].scale
        length = first_layer[p].length
        w1 = first_layer[p].input
        nugget = first_layer[p].nugget
        name = first_layer[p].name
        length = first_layer[p].length

        mu, var = gp(x=x_star, z=None, global_w1=None, w1=w1, Rinv=Rinv, Rinv_y=Rinv_y, 
                  scale=scale, length=length, nugget=nugget, name=name)
        mu_first_layer[:,p] = mu
        var_first_layer[:,p] = var
        lengths[p] = length[0]

        # dmu_dx
        d_k_one_vector_d_x_star = sexp_k_one_vector_derivative(x_star, w1, length, scale) # Shape: (M, N, D)
        d_k_one_vector_d_x_star_transpose = np.transpose(d_k_one_vector_d_x_star, (0, 2, 1)) # Shape: (M, D, N)
        dmu_dx[:,p,:,:] = np.einsum('mdn,n->md', d_k_one_vector_d_x_star_transpose, Rinv_y) # Shape: (M, D, 1)

        # dvar_dx
        second_der_kxx = second_k_xx_derivatives(length, scale, (M, D, D))
        quad_term = np.einsum('mdn,nk->mdk', d_k_one_vector_d_x_star_transpose, Rinv) # Shape: (M, D, N)
        quad_term = np.einsum('mdn,mnd->mdd', quad_term, d_k_one_vector_d_x_star) # Shape: (M, D, D)
        dvar_dx[:,p,:,:] = np.einsum("m,mdd->mdd", var, second_der_kxx - quad_term) # Shape: (M, D, D)

    w1 = second_layer[0].input
    # with shape (M,N,P) 
    # where M is the number of prediction points, 
    # N is the number of samples, 
    # P is the number of GP nodes
    epsilon_xstar, d_epsilon_dz_m, d_epsilon_dz_v \
    = epsilon_sexp(w1, mu_first_layer, var_first_layer, lengths, return_d_epsilon=True)
    for d in range(D):
        # partial derivative of I with respect to d-dimension x_star
        partial_I_partial_xd = np.zeros(N)
        for p in range(num_gp_nodes):
            # chain rule: dI/dx = dI/dz_m * dz_m/dx + dI/dz_v * dz_v/dx
            # shape of d_epsilon_dx_star: (M, N)
            first_term = d_epsilon_dz_m[:,:,p] * dmu_dx[:,p,d,0] + d_epsilon_dz_v[:,:,p] * dvar_dx[:,p,d,d]
            second_term = np.prod( np.delete( epsilon_xstar, p, axis=2 ) , axis=2) # shape: (M, N)
            partial_I_partial_xd += first_term * second_term
        nabla_I[:,:,d] = partial_I_partial_xd
    return nabla_I

def grad_lgp_sexp(x_star, all_layers):
    """Compute the gradient of the linked GP for the squared exponential kernel.
    input: x_star: the input point
           all_layers: all layers in the DGP model
    return: nabla_I: the gradient of the linked GP about x_star
    """
    nabla_I = nabla_sexp_I(x_star, all_layers)
    Rinv_y = all_layers[1][0].Rinv_y
    nabla_I_Rinv_y = np.einsum('mnd,n->md', nabla_I, Rinv_y)
    return nabla_I_Rinv_y


if __name__ == "__main__":
    from dgpsi import dgp, kernel, combine, lgp, path, emulator, gp
    from mogp_emulator.ExperimentalDesign import LatinHypercubeDesign
    import numpy as np

    # Define the benchmark function
    def test_f(x):
        x1, x2 = x[..., 0], x[..., 1]
        return np.sin(x1) * np.cos(x2) + np.sin(2 * x1) * np.cos(3 * x2)

    lhd = LatinHypercubeDesign([(-1,1),(-1,1)])

    x_train = lhd.sample(10)
    y_train = np.array(test_f(x_train))
    layer1=[kernel(length=np.array([1]),name='sexp', scale_est=True),kernel(length=np.array([1]),name='sexp', scale_est=False)]
    layer2=[kernel(length=np.array([1]),name='sexp', scale_est=True)]
    all_layer=combine(layer1,layer2)
    m=dgp(x_train,y_train[:,None],all_layer)
    m.train(N=150)
    final_layer_obj=m.estimate()
    emu=emulator(final_layer_obj,N=10)

    grid_eval_grid = np.linspace(-1, 1, 10)
    grid_eval_grid = np.meshgrid(grid_eval_grid, grid_eval_grid)
    grid_eval_grid = np.stack([grid_eval_grid[0].flatten(), grid_eval_grid[1].flatten()], axis=-1)

    grad_pred = grad_lgp_sexp(grid_eval_grid.reshape(-1,2), 
                              emu.all_layer_set[0])




            



            
            

    





































# def jax_gp_predict_sexp(x,layer):
#     """Make GP predictions.
#     """
#     Rinv = layer.Rinv
#     Rinv_y = layer.Rinv_y
#     scale = layer.scale
#     length = layer.length
#     w1 = layer.input
#     nugget = layer.nugget

#     r=jax_k_one_vec_sexp(w1,x,length)
#     Rinv_r=jnp.dot(Rinv,r)
#     r_Rinv_r=jnp.sum(r*Rinv_r,axis=0)
#     v=jnp.abs(scale*(1+nugget-r_Rinv_r)) 
#     m=jnp.dot(Rinv_y, r)
#     return m, v, length

# def cond_mean_sexp(x_star, all_layer):
#     first_layer = all_layer[0]
#     second_layer = all_layer[1]
#     mu_first_layer = jnp.zeros((1,len(first_layer)))
#     var_first_layer = jnp.zeros((1,len(first_layer)))
#     lengths = jnp.zeros(len(first_layer))
#     for p in range(len(first_layer)):
#         mu, var, length = jax_gp_predict_sexp(x_star, first_layer[p])
#         mu_first_layer = mu_first_layer.at[:,p].set(mu)
#         var_first_layer = var_first_layer.at[:,p].set(var)
#         lengths = lengths.at[p].set(length[0])
#     w1 = second_layer[0].input
#     # ms, vs, lengths = vmap_predict_x_and_layer(x_star, first_layer)
#     I = jax_I_sexp(w1, mu_first_layer, var_first_layer, lengths)
#     Rinv_y = second_layer[0].Rinv_y
#     IRinv_y=jnp.dot(I, Rinv_y) 
#     return IRinv_y



# def nabla_I_sexp(x_star, all_layer):
#     # X: **w** imputated random variable
#     # z_m: mu from last layer GP
#     # z_v: var from last layer GP
#     first_layer = all_layer[0]
#     second_layer = all_layer[1]

#     num_gp_nodes = len(first_layer)




 


# def jax_k_one_vec_sexp(X,z,length):
#     """Compute cross-correlation matrix between the testing and training input data.
#     """
#     X_l=X/length
#     z_l=z/length
#     L_X=jnp.expand_dims(jnp.sum(X_l**2,axis=1),axis=1)
#     L_z=jnp.sum(z_l**2,axis=1)
#     dis2=L_X-2*jnp.dot(X_l,z_l.T)+L_z
#     k=jnp.exp(-dis2)
#     return k

# def jax_gp_predict_sexp(x,layer):
#     """Make GP predictions.
#     """
#     Rinv = layer.Rinv
#     Rinv_y = layer.Rinv_y
#     scale = layer.scale
#     length = layer.length
#     w1 = layer.input
#     nugget = layer.nugget

#     r=jax_k_one_vec_sexp(w1,x,length)
#     Rinv_r=jnp.dot(Rinv,r)
#     r_Rinv_r=jnp.sum(r*Rinv_r,axis=0)
#     v=jnp.abs(scale*(1+nugget-r_Rinv_r)) 
#     m=jnp.dot(Rinv_y, r)
#     return m, v, length

# # @jax.jit
# def jax_I_sexp(X,z_m,z_v,length):
#     # X: w imputated random variable
#     # z_m: mu from last layer GP
#     # z_v: var from last layer GP
#     # n, d = X.shape
#     # I = jnp.zeros(n)
#     # X_z = X-z_m
#     # I_coef1 = 1.
#     # for k in range(d):
#     #     div = 2*z_v[:,k]/length[k]**2
#     #     I_coef1 *= 1 + div
#     # I_coef1 = 1./jnp.sqrt(I_coef1)
#     # for i in range(n):
#     #     I_coef2 = 0.
#     #     for k in range(d):
#     #         I_coef2 += X_z[i,k]**2/(2*z_v[:,k]+length[k]**2)
#     #     I_value = I_coef1 * jnp.exp(-I_coef2)
#     #     I = I.at[i].set( I_value[0] )
#     # return I



#     v_l=1/(1+2*z_v/length**2)
#     X_l=X/length
#     m_l=z_m/length
#     cross1=jnp.dot(X_l**2,v_l.T)
#     cross2=2*jnp.dot(X_l,(m_l*v_l).T)
#     L_z=jnp.sum(m_l**2*v_l,axis=1)
#     coef = 0.5*jnp.sum(jnp.log(v_l),axis=1)
#     dist=coef-cross1+cross2-L_z
#     I=jnp.exp(dist.T)
#     return I


# # def IJ_sexp(X, z_m, z_v, length, R2sexp, Psexp):
# #     n, d = X.shape
# #     I = np.zeros(n)
# #     J = np.zeros((n,n))
# #     X_z = X-z_m
# #     I_coef1, J_coef1 = 1., 1.
# #     for k in range(d):
# #         div = 2*z_v[k]/length[k]**2
# #         I_coef1 *= 1 + div
# #         J_coef1 *= 1 + 2*div
# #         J += (Psexp[k]-2*z_m[k]/length[k])**2/(2+4*div)
# #     I_coef1, J_coef1 = 1/sqrt(I_coef1), 1/sqrt(J_coef1)
# #     J = J_coef1 * np.exp(-J) * R2sexp
# #     for i in range(n):
# #         I_coef2 = 0.
# #         for k in range(d):
# #             I_coef2 += X_z[i,k]**2/(2*z_v[k]+length[k]**2)
# #         I[i] = I_coef1 * np.exp(-I_coef2)
# #     return I,J







# @jax.jit
# def jax_I_matern(X,z_m,z_v,length):
#     n, d = X.shape
#     I = jnp.zeros(n)
#     zX = z_m-X
#     muA, muB = zX-sqrt(5)*z_v/length, zX+sqrt(5)*z_v/length
#     for i in range(n):
#         Ii = 1.
#         for k in range(d):
#             if z_v[k]!=0:
#                 Ii *= np.exp((5*z_v[k]-2*sqrt(5)*length[k]*zX[i,k])/(2*length[k]**2))* \
#                     ((1+sqrt(5)*muA[i,k]/length[k]+5*(muA[i,k]**2+z_v[k])/(3*length[k]**2))*0.5*(1+erf(muA[i,k]/sqrt(2*z_v[k])))+ \
#                     (sqrt(5)+(5*muA[i,k])/(3*length[k]))*sqrt(0.5*z_v[k]/pi)/length[k]*np.exp(-0.5*muA[i,k]**2/z_v[k]))+ \
#                     np.exp((5*z_v[k]+2*sqrt(5)*length[k]*zX[i,k])/(2*length[k]**2))* \
#                     ((1-sqrt(5)*muB[i,k]/length[k]+5*(muB[i,k]**2+z_v[k])/(3*length[k]**2))*0.5*(1+erf(-muB[i,k]/sqrt(2*z_v[k])))+ \
#                     (sqrt(5)-(5*muB[i,k])/(3*length[k]))*sqrt(0.5*z_v[k]/pi)/length[k]*np.exp(-0.5*muB[i,k]**2/z_v[k]))
#             else:
#                 Ii *= (1+sqrt(5)*np.abs(zX[i,k])/length[k]+5*zX[i,k]**2/(3*length[k]**2))*np.exp(-sqrt(5)*np.abs(zX[i,k])/length[k])  
#         I = I.at[i].set(Ii)
#     return I




