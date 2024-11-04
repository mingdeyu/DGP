from numba import njit, set_num_threads, config, prange, vectorize
import numpy as np
from psutil import cpu_count
from dgpsi.vecchia import K_vec_nb
from dgpsi.functions import k_one_vec

core_num = cpu_count(logical = False)
config.THREADING_LAYER = 'workqueue'
set_num_threads(core_num)

@njit(cache=True, parallel=True)
def gp_pred(x,z,Rinv,Rinv_y,scale,length,nugget,name):
    """Make GP predictions
    """
    n_pred = x.shape[0]
    m, v = np.zeros(n_pred), np.zeros(n_pred)
    for i in prange(n_pred):
        ri=K_vec_nb(z,x[i],length,name)
        Rinv_ri=np.dot(Rinv,ri)
        r_Rinv_r=np.dot(ri, Rinv_ri)
        m[i] = np.dot(Rinv_y, ri)
        v[i] = np.abs(scale*(1+nugget-r_Rinv_r))[0]
    return m, v

@njit(cache=True)
def epsilon_sexp_with_derivative(x, z_m, z_v, length, scale):
    """ compute the epsilon function in the I function.
    N is the number of samples
    M is the number of prediction points
    D is the number of dimensions of the input
    P is the number of GP nodes in first layer

    input: x: the impuated random variable w (N, P) 
           z_m: the mu from the last layer GP (M, P) from x_star
           z_v: the var from the last layer GP (M, P) from x_star
           length: the length of the kernel (P,)
    return epsilon: the epsilon function in the I function (M, N, P)
           d_epsilon_dz_m: the derivative of epsilon with respect to z_m (M, N, P)
           d_epsilon_dz_v: the derivative of epsilon with respect to z_v (M, N, P)
    """
    N, P = x.shape
    M = z_m.shape[0]
    epsilon = np.zeros((M, N, P))
    d_epsilon_dz_m = np.zeros((M, N, P))
    d_epsilon_dz_v = np.zeros((M, N, P))

    for m in range(M):
        for n in range(N):
            for p in range(P):
                div = 1/np.sqrt(1+2*z_v[m, p]/length[p]**2)
                # div = 1/np.sqrt(1/length[p]**2)
                # div = 1
                exp_term = np.exp( -( z_m[m, p]-x[n, p] )**2 / ( 2*z_v[m, p]+length[p]**2 ))
                # exp_term = np.exp(-(z_m[m, p]-x[n, p])**2/(length[p]**2))
                epsilon[m, n, p] = div * exp_term   
                # print("epsilon: ", epsilon[m,n,p])
                d_epsilon_dz_m[m,n,p] = -2*(z_m[m,p]-x[n,p])/(2*z_v[m,p]+length[p]**2) * epsilon[m,n,p]
                # print("d_epsilon_dz_m: ", d_epsilon_dz_m[m,n,p])
                # # print("d_epsilon_dz_m: ", d_epsilon_dz_m[m,n,p])
                # # d_epsilon_dz_m[m,n,p] = -2*(z_m[m,p]-x[n,p])/(length[p]**2) * epsilon[m,n,p]
                d_epsilon_dz_v[m,n,p] = -(1/(2*z_v[m,p]+length[p]**2)-2*( (z_m[m,p]-x[n,p])/(2*z_v[m,p]+length[p]**2) )**2) * epsilon[m,n,p]
    return epsilon, d_epsilon_dz_m, d_epsilon_dz_v


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
    grad_k = - 2 * diff / length**2 * k_values[:, :, None]  # Broadcasting k_values to shape (M, N, D)
    return grad_k 



@njit(cache=True)
def matern_k_one_vector_derivative(x_star, X, length, scale):
    """
    Compute the derivative of the Matern-2.5 kernel with respect to x_star.

    Parameters:
    - x_star: ndarray of shape (M, D), the input points where the derivative is evaluated.
    - X: ndarray of shape (N, D), the training input points.
    - length: float, the length scale parameter (l).
    - scale: float, the signal variance (sigma^2).

    Returns:
    - grad_k: ndarray of shape (M, N, D), the gradients of the kernel.
    """
    # Ensure that x_star is a 2D array [M, D]
    assert x_star.ndim == 2, "x_star should be a 2D array with shape [M, D]"
    # Get the shapes
    M, D = x_star.shape
    N, _ = X.shape
    # Compute pairwise differences
    diff = x_star[:, None, :] - X[None, :, :]  # Shape: (M, N, D)
    # Compute squared distances and distances
    sq_dist = np.sum(diff ** 2, axis=-1)  # Shape: (M, N)
    r = np.sqrt(sq_dist + 1e-12)  # Add a small value to avoid division by zero
    # Compute constants
    sqrt_5 = np.sqrt(5.0)
    l = length
    # Compute the kernel values
    A = (1.0 + (sqrt_5 * r) / l + (5.0 * r ** 2) / (3.0 * l ** 2))
    exp_part = np.exp(-sqrt_5 * r / l)
    # k_values = A * exp_part  # Shape: (M, N)
    # Compute the derivative of the kernel with respect to r
    # Compute A_prime - (sqrt_5 / l) * A
    A_prime = (sqrt_5 / l) + (10.0 * r) / (3.0 * l ** 2)
    term = A_prime - (sqrt_5 / l) * A
    dk_dr = term * exp_part  # Shape: (M, N)
    # Compute the gradient
    grad_k = np.zeros_like(diff)  # Shape: (M, N, D)
    # Avoid division by zero
    inv_r = 1.0 / (r + 1e-12)
    # Compute gradient
    grad_k = (dk_dr * inv_r)[:, :, None] * diff  # Broadcasting to shape (M, N, D)
    return grad_k



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
    dvar_dx = np.zeros((M, num_gp_nodes, D, 1))

    for p in range(num_gp_nodes):
        Rinv = first_layer[p].Rinv
        Rinv_y = first_layer[p].Rinv_y
        scale = first_layer[p].scale[0]
        length = first_layer[p].length
        w1 = first_layer[p].input
        nugget = first_layer[p].nugget
        name = first_layer[p].name
        length = first_layer[p].length

        mu, var = gp_pred(x=x_star, z=w1, Rinv=Rinv, Rinv_y=Rinv_y, 
                  scale=scale, length=length, nugget=nugget, name=name)
        mu_first_layer[:,p] = mu
        var_first_layer[:,p] = var

        # dmu_dx
        if name == 'sexp':
            d_k_one_vector_d_x_star = sexp_k_one_vector_derivative(x_star, w1, length, scale)
        elif name == 'matern2.5':
            d_k_one_vector_d_x_star = matern_k_one_vector_derivative(x_star, w1, length, scale)
        else:
            raise ValueError("Only support squared exponential and Matern-2.5 kernel now.")
        d_k_one_vector_d_x_star = sexp_k_one_vector_derivative(x_star, w1, length, scale) # Shape: (M, N, D)
        d_k_one_vector_d_x_star_transpose = np.transpose(d_k_one_vector_d_x_star, (0, 2, 1)) # Shape: (M, D, N)
        dmu_dx[:,p,:,:] = np.einsum('mdn,n->md', d_k_one_vector_d_x_star_transpose, Rinv_y)[:,:,None] # Shape: (M, D, 1)

        term1 = -2*np.einsum('mdn,nk->mdk', d_k_one_vector_d_x_star_transpose, Rinv) # Shape: (M, D, N)
        # print(k_one_vec(x_star, w1, length, name).shape)
        term1 = np.einsum('mdn,mn->md', term1, k_one_vec(x_star, w1, length, name))[:,:,None]# Shape: (M, D, 1)
        dvar_dx[:,p,:,:] = scale*term1
    w1 = second_layer[0].input
    
    # ============ epsilon function ============
    epsilon_xstar, d_epsilon_dz_m, d_epsilon_dz_v = epsilon_sexp_with_derivative(w1, 
                                                    mu_first_layer, 
                                                    var_first_layer, 
                                                    np.array([second_layer[0].length[0] for i in range(num_gp_nodes)]),
                                                    scale=scale)
    for d in range(D):
        # partial derivative of I with respect to d-dimension x_star
        partial_I_partial_xd = np.zeros((M,N))
        s = np.zeros((M,N))
        for p in range(num_gp_nodes):
            # chain rule: dI/dx = dI/dz_m * dz_m/dx + dI/dz_v * dz_v/dx
            # shape of d_epsilon_dx_star: (M, N)
            first_term = np.einsum('mn,m->mn', d_epsilon_dz_m[:,:,p], dmu_dx[:,p,d,0]) # shape: (M, N) 
            first_term += np.einsum('mn,m->mn', d_epsilon_dz_v[:,:,p], dvar_dx[:,p,d,0])

            second_term = np.prod( np.delete( epsilon_xstar, p, axis=2 ) , axis=2) # shape: (M, N)
            partial_I_partial_xd += first_term * second_term

        nabla_I[:,:,d] = partial_I_partial_xd

    return nabla_I

def grad_lgp(x_star, all_layers):
    """Compute the gradient of the linked GP for the squared exponential kernel.
    input: x_star: the input point
           all_layers: all layers in the DGP model
    return: nabla_I: the gradient of the linked GP about x_star
    """
    assert len(all_layers) == 2, "Only support two layers now."
    assert len(all_layers[1]) == 1, "Only support one GP in the second layer now."
    assert all_layers[1][0].name == 'sexp', "Only support squared exponential kernel now."

    nabla_I = nabla_sexp_I(x_star, all_layers)
    Rinv_y = all_layers[1][0].Rinv_y
    nabla_I = np.transpose(nabla_I, axes=(0,2,1))

    nabla_I_Rinv_y = np.einsum('mdn,n->md', nabla_I, Rinv_y)
    return nabla_I_Rinv_y


if __name__ == "__main__":
    from dgpsi import dgp, kernel, combine, lgp, path, emulator, gp, nb_seed
    from mogp_emulator.ExperimentalDesign import LatinHypercubeDesign
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(123)
    nb_seed(123)
    

    # Define the benchmark function
    def test_f(x):
        x1, x2 = x[..., 0], x[..., 1]
        return 3*np.sin(x1) * np.cos(x2) + 8 * np.sin(2 * x1) * np.cos(3 * x2) 

    # Define the closed-form gradient function
    def gradient_test_f(x):
        x1, x2 = x[..., 0], x[..., 1]
        df_dx = 3 * np.cos(x1) * np.cos(x2) + 8 * 2 * np.cos(2 * x1) * np.cos(3 * x2) 
        df_dy = -3 * np.sin(x1) * np.sin(x2) - 8 * 3 * np.sin(2 * x1) * np.sin(3 * x2) 
        return np.array([df_dx, df_dy])

    x_train = np.linspace(-1, 1, 5)
    x_train = np.meshgrid(x_train, x_train)
    x_train = np.stack([x_train[0].flatten(), x_train[1].flatten()], axis=-1)

    # x_train = lhd.sample(64)
    y_train = np.array(test_f(x_train))
    layer1=[kernel(length=np.array([1]),name='sexp', scale_est=True),
            kernel(length=np.array([1]),name='sexp', scale_est=True)]
    layer2=[kernel(length=np.array([1]),name='sexp', scale_est=True)]
    all_layer=combine(layer1,layer2)
    m=dgp(x_train,y_train[:,None],all_layer)
    m.train(N=100)
    final_layer_obj=m.estimate()
    emu=emulator(final_layer_obj,N=50)
    # gp_emu = gp(x_train, y_train, kernel(length=np.array([1]), name='sexp', scale_est=True))
    # gp_emu.train()
    grid_eval_func = np.linspace(-1, 1, 50)
    grid_eval_func = np.meshgrid(grid_eval_func, grid_eval_func)
    grid_eval_func = np.stack([grid_eval_func[0].flatten(), grid_eval_func[1].flatten()], axis=-1)
    Z = test_f(grid_eval_func)  
    pred_mu, _ = emu.predict(grid_eval_func)


    grid_eval_grid = np.linspace(-1, 1, 10)
    grid_eval_grid = np.meshgrid(grid_eval_grid, grid_eval_grid)
    grid_eval_grid = np.stack([grid_eval_grid[0].flatten(), grid_eval_grid[1].flatten()], axis=-1)
    grad_eval = gradient_test_f(grid_eval_grid)


    grad_pred = np.zeros((100, 2))
    num_imp = len(emu.all_layer_set)
    for i in range(num_imp):
        tmp_grad_pred = grad_lgp(grid_eval_grid, emu.all_layer_set[i])
        grad_pred += tmp_grad_pred/num_imp


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].quiver(grid_eval_grid[:, 0], grid_eval_grid[:, 1], grad_eval[0], grad_eval[1])
    colorbar_1 = ax[0].imshow(Z.reshape(50,50), extent=(-1, 1, -1, 1))
    fig.colorbar(colorbar_1, ax=ax[0])
    ax[0].set_title("Test function", fontsize=15)
    ax[0].set_xlabel("x1", fontsize=12)
    ax[0].set_ylabel("x2", fontsize=12)
    ax[0].grid()

    ax[1].quiver(grid_eval_grid[:, 0], grid_eval_grid[:, 1], grad_pred[:, 0], grad_pred[:, 1])
    colorbar_2 = ax[1].imshow(pred_mu.reshape(50, 50), extent=(-1, 1, -1, 1))
    fig.colorbar(colorbar_2, ax=ax[1])
    ax[1].set_title("Emulator prediction", fontsize=15)
    ax[1].set_xlabel("x1", fontsize=12)
    ax[1].set_ylabel("x2", fontsize=12)
    ax[1].scatter(x_train[:, 0], x_train[:, 1], color='red', marker='x')
    ax[1].grid()
    plt.show()

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # =============
    # First subplot
    # =============
    # set up the Axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(grid_eval_func[:, 0].reshape(50, 50), grid_eval_func[:, 1].reshape(50, 50), Z.reshape(50, 50), cmap='viridis')
    ax.set_title("Test function", fontsize=15)
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_zlabel("f(x1, x2)", fontsize=12)
    ax.view_init(30, 30)

    # ==============
    # Second subplot
    # ==============
    # set up the Axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(grid_eval_func[:, 0].reshape(50, 50), grid_eval_func[:, 1].reshape(50, 50), pred_mu.reshape(50, 50), cmap='viridis')
    ax.set_title("Emulator prediction", fontsize=15)
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_zlabel("f(x1, x2)", fontsize=12)
    ax.view_init(30, 30)


    plt.show()




            



            
            

    





































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




