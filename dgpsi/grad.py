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

def grad_gp(x_star, emu, return_variance=True):
    D = x_star.shape[1]
    from dgpsi import gp 
    """Compute the gradient of the GP emulator."""
    assert isinstance(emu, gp), "The input emulator should be a gp object."
    if emu.kernel.name == 'sexp':
        nabla_r = sexp_k_one_vector_derivative(x_star, emu.X, emu.kernel.length)
    elif emu.kernel.name == 'matern2.5':
        nabla_r = matern_k_one_vector_derivative(x_star, emu.X, emu.kernel.length)

    if return_variance == False:
        grad_pred_mu = np.einsum('mnd,n->md', nabla_r, emu.kernel.Rinv_y)
        return grad_pred_mu
    else:
        grad_pred_mu = np.einsum('mnd,n->md', nabla_r, emu.kernel.Rinv_y)
        grad_pred_var = np.einsum('mdn,nk->mdk', np.transpose(nabla_r, (0,2,1)), emu.kernel.Rinv)
        grad_pred_var = np.einsum('mkn,mnd->mkd', grad_pred_var, nabla_r)
        grad_pred_var = grad_pred_var[:,:,0] if D == 1 else grad_pred_var
        grad_pred_var = np.sqrt(np.clip(grad_pred_var, a_min=1e-6, a_max=None))
        return grad_pred_mu, grad_pred_var

@njit(cache=True)
def epsilon_sexp_with_derivative(x, z_m, z_v, length):
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
                exp_term = np.exp( -( z_m[m, p]-x[n, p] )**2 / ( 2*z_v[m, p]+length[p]**2 ))
                epsilon[m, n, p] = div * exp_term   
                d_epsilon_dz_m[m,n,p] = -2*(z_m[m,p]-x[n,p])/(2*z_v[m,p]+length[p]**2) * epsilon[m,n,p]
                d_epsilon_dz_v[m,n,p] = -(1/(2*z_v[m,p]+length[p]**2)-2*( (z_m[m,p]-x[n,p])/(2*z_v[m,p]+length[p]**2) )**2) * epsilon[m,n,p]
    return epsilon, d_epsilon_dz_m, d_epsilon_dz_v


@njit(cache=True)
def sexp_k_one_vector_derivative(x_star, X, length):
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

sqrt_5 = np.sqrt(5.0)
@njit(cache=True)
def matern_k_one_vector_derivative(x_star, X, length):
    """
    Compute the derivative of the Matern-2.5 kernel with respect to x_star.

    Parameters:
    - x_star: ndarray of shape (M, D), the input points where the derivative is evaluated.
    - X: ndarray of shape (N, D), the training input points.
    - length: float, the length scale parameter (l).

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

    


def nabla_sexp_I(x_star:np.array, all_layers:list, return_variance=True):
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
    # var_dx_1 = np.zeros((M, num_gp_nodes, D))

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
            d_k_one_vector_d_x_star = sexp_k_one_vector_derivative(x_star, w1, length)
        elif name == 'matern2.5':
            d_k_one_vector_d_x_star = matern_k_one_vector_derivative(x_star, w1, length)
        else:
            raise ValueError("Only support squared exponential and Matern-2.5 kernel now.")
        
        d_k_one_vector_d_x_star_transpose = np.transpose(d_k_one_vector_d_x_star, (0, 2, 1)) # Shape: (M, D, N)
        dmu_dx[:,p,:,:] = np.einsum('mdn,n->md', d_k_one_vector_d_x_star_transpose, Rinv_y)[:,:,None] # Shape: (M, D, 1)

        term1 = -2*np.einsum('mdn,nk->mdk', d_k_one_vector_d_x_star_transpose, Rinv) # Shape: (M, D, N)
        term1 = np.einsum('mdn,mn->md', term1, k_one_vec(x_star, w1, length, name))[:,:,None]# Shape: (M, D, 1)
        dvar_dx[:,p,:,:] = scale*term1

    w1 = second_layer[0].input
    scale = second_layer[0].scale[0]
    
    # ============ epsilon function ============
    epsilon_xstar, d_epsilon_dz_m, d_epsilon_dz_v = epsilon_sexp_with_derivative(w1, 
                                                    mu_first_layer, 
                                                    var_first_layer, 
                                                    np.array([second_layer[0].length[0] for _ in range(num_gp_nodes)]))
    
    # approxiamte the variance of the deep GP
    # Var(\nabla f_l(x_star)) = (1) \nabla f_{l-1}(x_star)^T * Var(f_l(x_star)) * \nabla f_{l-1}(x_star) 
    #                           (2) + Var(\nabla f_{l}(w_star|x_star))
    # (1)
    # shape of dmu_dx: (M, Num_gp_nodes, D, 1)
    dmu_dx_T = np.transpose(dmu_dx, axes=(0, 2, 1, 3)) # shape: (M, D, Num_gp_nodes, 1)

    nabla_rw = sexp_k_one_vector_derivative(mu_first_layer, w1, second_layer[0].length) # Shape: (M, N, G)
    nabla_rw_T = np.transpose(nabla_rw, axes=(0, 2, 1)) # Shape: (M, G, N)

    for d in range(D):
        # partial derivative of I with respect to d-dimension x_star
        partial_I_partial_xd = np.zeros((M,N))
        for p in range(num_gp_nodes):
            # chain rule: dI/dx = dI/dz_m * dz_m/dx + dI/dz_v * dz_v/dx
            # shape of d_epsilon_dx_star: (M, N)
            first_term = np.einsum('mn,m->mn', d_epsilon_dz_m[:,:,p], dmu_dx[:,p,d,0]) # shape: (M, N) 
            first_term += np.einsum('mn,m->mn', d_epsilon_dz_v[:,:,p], dvar_dx[:,p,d,0])

            second_term = np.prod( np.delete( epsilon_xstar, p, axis=2 ) , axis=2) # shape: (M, N)
            partial_I_partial_xd += first_term * second_term

        nabla_I[:,:,d] = partial_I_partial_xd

    if return_variance:
        var_dx_2 = scale/length**4 if length > 1 else scale
        quad_term = np.einsum('mgn,nk->mgk', nabla_rw_T, second_layer[0].Rinv) # shape: (M, G, N)
        quad_term = np.einsum('mgn,mnk->mgk', quad_term, nabla_rw) # shape: (M, G, G)
        # print(quad_term.mean())
        var_dx_2 = var_dx_2 + quad_term

        V_nabla_f = np.einsum('mdg, mgk->mdk', dmu_dx_T[:,:,:,0], var_dx_2) # shape: (M, D, G)
        V_nabla_f = np.einsum('mdg, mgk->mdk', V_nabla_f, dmu_dx[:,:,:,0]) # shape: (M, D)

        return nabla_I, V_nabla_f
    else:
        return nabla_I

def nabla_matern_I(x_star, all_layers, return_variance=True):
    """Compute the nabla of the I function for the Matern-2.5 kernel.
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
    # nabla_I = np.zeros((M, N, D))

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


        mu, var = gp_pred(x=x_star, z=w1, Rinv=Rinv, Rinv_y=Rinv_y, 
                  scale=scale, length=length, nugget=nugget, name=name)
        mu_first_layer[:,p] = mu
        var_first_layer[:,p] = var
    
        # dmu_dx
        if name == 'sexp':
            d_k_one_vector_d_x_star = sexp_k_one_vector_derivative(x_star, w1, length)
        elif name == 'matern2.5':
            d_k_one_vector_d_x_star = matern_k_one_vector_derivative(x_star, w1, length)
        else:
            raise ValueError("Only support squared exponential and Matern-2.5 kernel now.")
        
        d_k_one_vector_d_x_star_transpose = np.transpose(d_k_one_vector_d_x_star, (0, 2, 1)) # Shape: (M, D, N)
        dmu_dx[:,p,:,:] = np.einsum('mdn,n->md', d_k_one_vector_d_x_star_transpose, Rinv_y)[:,:,None] # Shape: (M, D, 1)

        term1 = -2*np.einsum('mdn,nk->mdk', d_k_one_vector_d_x_star_transpose, Rinv) # Shape: (M, D, N)
        term1 = np.einsum('mdn,mn->md', term1, k_one_vec(x_star, w1, length, name))[:,:,None] # Shape: (M, D)
        dvar_dx[:,p,:,:] = scale*term1


    w = second_layer[0].input
    scale = second_layer[0].scale[0]
    length = second_layer[0].length[0]
    Rinv_y = second_layer[0].Rinv_y
    
    d_k_one_vector_d_w_star = matern_k_one_vector_derivative(mu_first_layer, 
                                                             w, length)
    d_k_one_vector_d_w_star_transpose = np.transpose(d_k_one_vector_d_w_star, axes=(0, 2, 1)) # shape: (M, D, N)
    dmu2_dmu1 = np.einsum('mdn,n->md', d_k_one_vector_d_w_star_transpose, Rinv_y) # shape: (M, Num_gp_nodes)
    dmu2_dx = np.einsum('mp,mpd->md', dmu2_dmu1, dmu_dx[:,:,:,0]) # shape: (M, D)

    if return_variance:
        var2_dw = np.einsum('mdn,nk->mdk', d_k_one_vector_d_w_star_transpose, second_layer[0].Rinv) # shape: (M, D, N)
        var2_dw = np.einsum('mdn,mnk->mdk', var2_dw, d_k_one_vector_d_w_star) # shape: (M, D, D)

        var2_dx = np.einsum('mdg, mgk->mdk', np.transpose(dmu_dx[:,:,:,0],axes=(0, 2, 1)), var2_dw) # shape: (M, D, D)
        var2_dx = np.einsum('mdg, mgk->mdk', var2_dx, dmu_dx[:,:,:,0]) * (1/num_gp_nodes**D) # shape: (M, D)

        return dmu2_dx, var2_dx
    else:
        return dmu2_dx



def grad_lgp(x_star, all_layers, return_variance=True):
    """Compute the gradient of the linked GP for the squared exponential kernel.
    input: x_star: the input point
           all_layers: all layers in the DGP model
    return: nabla_I: the gradient of the linked GP about x_star
    """
    assert len(all_layers) == 2, "Only support two layers now."
    assert len(all_layers[1]) == 1, "Only support one GP in the second layer now."


    if all_layers[1][0].name == 'sexp':
        if return_variance:
            nabla_I, V_nabla_f = nabla_sexp_I(x_star, all_layers, return_variance=return_variance)
            Rinv_y = all_layers[1][0].Rinv_y
            nabla_I = np.transpose(nabla_I, axes=(0,2,1))

            nabla_I_Rinv_y = np.einsum('mdn,n->md', nabla_I, Rinv_y)

            return nabla_I_Rinv_y, V_nabla_f
        else:
            nabla_I = nabla_sexp_I(x_star, all_layers, return_variance=return_variance)
            Rinv_y = all_layers[1][0].Rinv_y
            nabla_I = np.transpose(nabla_I, axes=(0,2,1))
            nabla_I_Rinv_y = np.einsum('mdn,n->md', nabla_I, Rinv_y)

            return nabla_I_Rinv_y
    
    elif all_layers[1][0].name == 'matern2.5':
        return nabla_matern_I(x_star, all_layers, return_variance=return_variance)
    
    else:
        raise ValueError("Only support squared exponential and Matern-2.5 kernel now.")
    
# def grad_dgp(x_star, emu, return_variance=True):
#     N, D = x_star.shape
#     num_imp = len(emu.all_layer_set)
    
#     # Initialize arrays using `np.zeros_like` where possible for clarity
#     grad_pred = np.zeros_like(x_star)

#     # Efficient accumulation in a loop
#     if return_variance:
#         grad_pred_var = np.zeros((N, D, D))
#         grad_pred_var_imp_list = []
#         grad_pred_mu_imp_list = []
#         for layer in emu.all_layer_set:
#             tmp_grad_pred, tmp_grad_pred_var = grad_lgp(x_star, layer, return_variance=True)
#             grad_pred += tmp_grad_pred / num_imp
#             grad_pred_mu_imp_list.append(tmp_grad_pred)
#             grad_pred_var_imp_list.append(tmp_grad_pred_var)
#         for i in range(len(grad_pred_var_imp_list)):
#             grad_pred_var += (1/num_imp)*grad_pred_var_imp_list[i] + (1/num_imp)*(grad_pred_mu_imp_list[i]@grad_pred_mu_imp_list[i].T)
#         grad_pred_var = grad_pred_var - grad_pred@grad_pred.T
#     else:
#         for layer in emu.all_layer_set:
#             tmp_grad_pred = grad_lgp(x_star, layer, return_variance=False)
#             grad_pred += tmp_grad_pred / num_imp
#         return grad_pred


#     # for layer in emu.all_layer_set:
#     #     tmp_grad_pred, tmp_grad_pred_var = grad_lgp(x_star, layer, )
#     #     grad_pred += tmp_grad_pred / num_imp
#     #     grad_pred_var_imp_list.append(tmp_grad_pred_var) if return_variance else None
#     #     if return_variance:
#     #         grad_pred_var += tmp_grad_pred_var / num_imp 

#     return (grad_pred, grad_pred_var) if return_variance else grad_pred


def grad_dgp(x_star, emu, return_variance=True):
    """
    Gradient prediction for a Deep Gaussian Process (DGP).
    
    Parameters
    ----------
    x_star : np.ndarray, shape (N, D)
        Test inputs.
    emu : object
        Emulator object containing all_layer_set.
    return_variance : bool, default=True
        Whether to return gradient variance.
    
    Returns
    -------
    grad_pred : np.ndarray, shape (N, D)
        Mean gradient prediction.
    grad_pred_var : np.ndarray, shape (N, D, D), optional
        Gradient covariance prediction, if return_variance=True.
    """
    N, D = x_star.shape
    num_imp = len(emu.all_layer_set)

    # Mean gradient accumulator
    grad_pred = np.zeros((N, D))

    if return_variance:
        # Expectation of covariance
        grad_pred_var = np.zeros((N, D, D))

        grad_pred_mu_imp_list = []
        for layer in emu.all_layer_set:
            mu_grad, cov_grad = grad_lgp(x_star, layer, return_variance=True)
            grad_pred += mu_grad / num_imp
            grad_pred_mu_imp_list.append(mu_grad)
            grad_pred_var += cov_grad / num_imp

        # Covariance of expectation
        mu_outer_sum = np.zeros((N, D, D))
        for i in range(len(grad_pred_mu_imp_list)):
            mu_grad = grad_pred_mu_imp_list[i]
            mu_outer_sum += np.einsum("ni,nj->nij", mu_grad, mu_grad) / num_imp
        grad_pred_var += mu_outer_sum
        grad_pred_var -= np.einsum("ni,nj->nij", grad_pred, grad_pred)
        return grad_pred, grad_pred_var
    else:
        for layer in emu.all_layer_set:
            mu_grad = grad_lgp(x_star, layer, return_variance=False)
            grad_pred += mu_grad / num_imp
        return grad_pred



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
    grad_pred_var = np.zeros((100, 2, 2))
    num_imp = len(emu.all_layer_set)
    for i in range(num_imp):
        tmp_grad_pred, tmp_grad_pred_var = grad_lgp(grid_eval_grid, emu.all_layer_set[i])
        grad_pred += tmp_grad_pred/num_imp
        grad_pred_var += tmp_grad_pred_var/(10*num_imp)


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
            