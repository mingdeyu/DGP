from numpy.random import uniform
import numpy as np
from scipy.linalg import cholesky
from math import sqrt
from .functions import update_f, fmvn, mvn_from_chol_nb
from .vecchia import fmvn_sp, nn_fwd_from_rev, L_matrix_nb, fmvn_sp_nb, U_matrix_sp_nb #,U_matrix_sp

class imputer:
    """Class to implement imputation of latent variables.

    Args:
        all_layer (list): a list that contains the DGP model
        block (bool, optional): whether to use the blocked (layer-wise) ESS for the imputations. Defaults to `True`.
    """
    def __init__(self, all_layer, block=True):
        self.all_layer=all_layer
        self.block=block
        self._build_cache()

    def __setstate__(self, state):
        if 'block' not in state:
            state['block'] = True
        self.__dict__.update(state)

        if not hasattr(self, "_upper_map") or not hasattr(self, "_is_hetero"):
            self._build_cache()

    def _build_cache(self):
        n_layer = len(self.all_layer)
        self._is_hetero = [False] * (n_layer - 1)
        self._upper_map = [None] * (n_layer - 1)

        for l in range(n_layer - 1):
            layer = self.all_layer[l]
            upper = self.all_layer[l + 1]

            self._is_hetero[l] = any(
                (k.type == 'likelihood' and k.exact_post_idx is not None) for k in upper
            )

            m = len(layer)
            mp = [[] for _ in range(m)]
            for uk in upper:
                for idx in uk.input_dim:
                    mp[idx].append(uk)
            self._upper_map[l] = mp

    def sample(self,burnin=0):
        """Implement the imputation via the ESS-within-Gibbs.

        Args:
            burnin (int, optional): the number of burnin iterations for the ESS-within-Gibbs sampler
                to generate one realisation of latent variables. Defaults to `0`.
        """
        all_layer = self.all_layer
        n_layer = len(all_layer)
        is_hetero = self._is_hetero
        upper_map = self._upper_map
        block = self.block
        L0 = None
        sqrt_s0 = None
        V0 = None
        if block:
            layer0 = all_layer[0]
            m0 = len(layer0)
            L0 = [None] * m0
            sqrt_s0 = [0.0] * m0
            V0 = [None] * m0
            for i, k in enumerate(layer0):
                sqrt_s0[i] = sqrt(float(k.scale[0]))
                if not k.vecch:
                    K = k.k_matrix()  # unscaled correlation(+nugget)
                    L0[i] = cholesky(K, lower=True, check_finite=False)
                else:
                    # Build ordered X once for this sample() call
                    if k.global_input is not None:
                        X = np.concatenate((k.input, k.global_input), 1)
                    else:
                        X = k.input
                    NN_rev, NN_count = k.NN_rev, k.NN_count
                    NN_fwd = nn_fwd_from_rev(NN_rev, NN_count)
                    Lmat = L_matrix_nb(X[k.ord], NN_rev, NN_count, k.length, k.nugget[0], k.name)
                    V0[i] = (Lmat, NN_fwd, NN_count)
        for _ in range(burnin+1):
            for l in range(n_layer-1):
                layer = all_layer[l]
                linked_layer = all_layer[l+1]
                if block and (not is_hetero[l]):
                    if l == 0:
                        self.one_sample_block(layer, linked_layer, L0, sqrt_s0, V0)  # cached chol(K) for layer 0
                    else:
                        self.one_sample_block(layer, linked_layer)
                else:
                    mp = upper_map[l]
                    for k, target_kernel in enumerate(layer):
                        self.one_sample(target_kernel, mp[k], k)

    @staticmethod
    def one_sample_block(target_layer,upper_layer, L_list=None, sqrt_s_list=None, V_list=None):
        """Impute a latent layer.

        Args:
            target_layer (list): a list of GPs that produce a latent layer that needs to be imputed.
            upper_layer (list): a list of GPs (in the next layer) that are fed by the output of GPs in **target_layer**.
        """
        M, N = len(target_layer), len(target_layer[0].output)
        f, nu = np.empty((N,M)), np.empty((N,M))
        for i, kernel in enumerate(target_layer):
            f[:,i] = kernel.output[:, 0]
            if kernel.vecch:
                if V_list is not None and V_list[i] is not None:
                    Lmat, NN_fwd, NN_count = V_list[i]
                    tmp = fmvn_sp_nb(Lmat, NN_fwd, NN_count, sqrt_s_list[i])
                    np.take(tmp, kernel.rev_ord, out=nu[:, i])
                else:
                    if kernel.global_input is not None:
                        X=np.concatenate((kernel.input, kernel.global_input),1)
                    else:
                        X=kernel.input
                    tmp = fmvn_sp(X[kernel.ord], kernel.NN_rev, kernel.NN_count, kernel.length, kernel.nugget[0], kernel.scale[0], kernel.name)
                    np.take(tmp, kernel.rev_ord, out=nu[:, i])
            else:
                if L_list is not None and L_list[i] is not None:
                    # use cached chol(K) for layer-0 kernels
                    nu[:, i] = mvn_from_chol_nb(L_list[i], sqrt_s_list[i])
                else:
                    # fallback (original behavior)
                    nu[:,i] = fmvn(kernel.k_matrix(), kernel.scale)

        #f = np.vstack([kernel.output.flatten() for kernel in target_layer]).T
        # Choose the ellipse for this sampling iteration.
        #nu = np.random.default_rng().multivariate_normal(mean=np.zeros(len(f)),cov=covariance,check_valid='ignore')     
        #nu = np.vstack([fmvn(kernel.scale*kernel.k_matrix()) for kernel in target_layer]).T            
        # Set the candidate acceptance threshold.
        log_y=0
        for linked_kernel in upper_layer:
            if linked_kernel.type=='gp':
                if linked_kernel.vecch:
                    log_y += linked_kernel.log_likelihood_func_vecch()
                else:      
                    log_y += linked_kernel.log_likelihood_func()
            elif linked_kernel.type=='likelihood': 
                log_y += linked_kernel.llik()
        log_y += np.log(uniform())
        # Set the bracket for selecting candidates on the ellipse.
        theta = uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Iterates until an candidate is selected.
        while True:
            # Generates a point on the ellipse defines by `nu` and the input. We
            # also compute the log-likelihood of the candidate and compare to
            # our threshold.
            #iter_count += 1
            fp = update_f(f,nu,theta)
            log_yp=0
            for linked_kernel in upper_layer:
                if linked_kernel.rep is None:
                    linked_kernel.input=fp[:,linked_kernel.input_dim]
                else:
                    if linked_kernel.type=='gp':
                        linked_kernel.input=fp[:,linked_kernel.input_dim]
                    else:
                        linked_kernel.input=fp[linked_kernel.rep,:][:,linked_kernel.input_dim]
                if linked_kernel.type=='gp':
                    if linked_kernel.vecch:
                        log_yp += linked_kernel.log_likelihood_func_vecch()
                    else:
                        log_yp += linked_kernel.log_likelihood_func()
                elif linked_kernel.type=='likelihood': 
                    log_yp += linked_kernel.llik()
            if log_yp > log_y:
                for k in range(M):
                    target_layer[k].output[:,0]=fp[:,k]
                return
            else:
                # If the candidate is not selected, shrink the bracket and
                # generate a new `theta`, which will yield a new candidate
                # point on the ellipse.
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = uniform(theta_min, theta_max)
    
    @staticmethod
    def one_sample(target_kernel,linked_upper_kernels,k):
        """Impute one latent variable produced by a particular GP.

        Args:
            target_kernel (class): the GP whose output is a latent variable that needs to be imputed.
            linked_upper_kernels (list): a list of GPs (in the next layer) that link the output produced
                by the GP defined by the argument **target_kernel**.
            k (int): the index indicating the position of the GP defined by the argument **target_kernel** in
                its layer.
        """
        if target_kernel.vecch:
            if target_kernel.global_input is not None:
                X=np.concatenate((target_kernel.input, target_kernel.global_input),1)
            else:
                X=target_kernel.input
            covariance = None
        else:
            covariance=target_kernel.k_matrix()
                  
        if (len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx is not None):
            lk = linked_upper_kernels[0]
            idx=np.where(lk.input_dim == k)[0][0]
            if idx in lk.exact_post_idx:
                if target_kernel.vecch:
                    if lk.rep is not None:
                        ord =  target_kernel.ord               
                        invGamma = 1.0/np.exp(lk.input[:,1])
                        invd = 1/(np.bincount(lk.rep, weights=invGamma, minlength=X.shape[0])[ord])
                        U_sp_latent, U_sp_obs_latent = U_matrix_sp_nb(X[ord], target_kernel.imp_NNarray, target_kernel.scale[0], target_kernel.length, 0.0, target_kernel.name, invd, target_kernel.imp_pointer_row, target_kernel.imp_pointer_col)
                        f=lk.posterior_vecch(idx=idx, U_sp_l=U_sp_latent, U_sp_ol=U_sp_obs_latent, ord=ord, rev_ord=target_kernel.rev_ord, invd=invd, invg=invGamma)
                        #U_sp_latent, U_sp_obs_latent= U_matrix_sp_rep(X[target_kernel.ord], target_kernel.imp_NNarray, target_kernel.rep_hetero, target_kernel.ord, target_kernel.scale[0], target_kernel.length, target_kernel.nugget[0], target_kernel.name, Gamma, target_kernel.imp_pointer_row, target_kernel.imp_pointer_col)
                        #f = linked_upper_kernels[0].posterior_vecch(idx=idx, U_sp_l=U_sp_latent, U_sp_ol=U_sp_obs_latent, ord=target_kernel.ord, rev_ord=target_kernel.rev_ord)    
                    else:
                        ord =  target_kernel.ord
                        Gamma = np.exp(lk.input[:,1])[ord]
                        U_sp_latent, U_sp_obs_latent = U_matrix_sp_nb(X[ord], target_kernel.imp_NNarray, target_kernel.scale[0], target_kernel.length, 0.0, target_kernel.name, Gamma, target_kernel.imp_pointer_row, target_kernel.imp_pointer_col)
                        f=lk.posterior_vecch(idx=idx, U_sp_l=U_sp_latent, U_sp_ol=U_sp_obs_latent, ord=ord, rev_ord=target_kernel.rev_ord)
                else:
                    #np.fill_diagonal(covariance, target_kernel.scale)
                    f=lk.posterior(idx=idx,v=target_kernel.scale * covariance)
                if lk.rep is None:
                    lk.input[:,idx]=f
                else:
                    lk.input[:,idx]=f[lk.rep]
                target_kernel.output[:,0]=f
                return
        
        f = target_kernel.output[:,0]
        # Choose the ellipse for this sampling iteration.
        #nu = np.random.default_rng().multivariate_normal(mean=np.zeros(len(f)),cov=covariance,check_valid='ignore')  
        if target_kernel.vecch:
            nu = fmvn_sp(X[target_kernel.ord], target_kernel.NN_rev, target_kernel.NN_count, target_kernel.length, target_kernel.nugget[0], target_kernel.scale[0], target_kernel.name)[target_kernel.rev_ord]
        else:
            nu = fmvn(covariance, target_kernel.scale)                       
        # Set the candidate acceptance threshold.
        log_y=0
        for linked_kernel in linked_upper_kernels:
            if linked_kernel.type=='gp':
                if linked_kernel.vecch:
                    log_y += linked_kernel.log_likelihood_func_vecch()
                else:      
                    log_y += linked_kernel.log_likelihood_func()
            elif linked_kernel.type=='likelihood': 
                log_y += linked_kernel.llik()
        log_y += np.log(uniform())
        # Set the bracket for selecting candidates on the ellipse.
        theta = uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Iterates until an candidate is selected.
        while True:
            # Generates a point on the ellipse defines by `nu` and the input. We
            # also compute the log-likelihood of the candidate and compare to
            # our threshold.
            fp = update_f(f,nu,theta)
            log_yp=0
            for linked_kernel in linked_upper_kernels:
                pos = np.where(linked_kernel.input_dim==k)[0][0]
                if linked_kernel.rep is None:
                    linked_kernel.input[:,pos]=fp
                else:
                    if linked_kernel.type=='gp':
                        linked_kernel.input[:,pos]=fp
                    else:
                        linked_kernel.input[:,pos]=fp[linked_kernel.rep]
                if linked_kernel.type=='gp':
                    if linked_kernel.vecch:
                        log_yp += linked_kernel.log_likelihood_func_vecch()
                    else:
                        log_yp += linked_kernel.log_likelihood_func()
                elif linked_kernel.type=='likelihood': 
                    log_yp += linked_kernel.llik()
            if log_yp > log_y:
                target_kernel.output[:,0]=fp
                return
            else:
                # If the candidate is not selected, shrink the bracket and
                # generate a new `theta`, which will yield a new candidate
                # point on the ellipse.
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = uniform(theta_min, theta_max)
    
    def key_stats(self):
        """Compute and store key statistics used in predictions
        """
        n_layer=len(self.all_layer)
        for l in range(n_layer):
            layer=self.all_layer[l]
            for kernel in layer:
                if kernel.type == 'gp':
                    kernel.compute_stats()

    def update_ord_nn(self):
        """
        Efficient + simple update:
        - Use dict lookup (O(K)) instead of scanning previous kernels (O(K^2))
        - If match exists: pass ord/NN/rev_ord/NN_rev/NN_count into ord_nn() to reuse
        - pointer is still per-kernel (as in your original code)
        """
        n_layer = len(self.all_layer)

        for l in range(n_layer):
            layer = self.all_layer[l]
            rep = {}  # key -> representative kernel already processed

            for kernel in layer:
                if kernel.type != "gp":
                    continue

                # keep your original semantics:
                compute_pointer = False if kernel.imp_pointer_row is None else True

                input_key = tuple(kernel.input_dim.tolist())
                conn = kernel.connect
                conn_key = None if conn is None else tuple(conn.tolist())

                if len(kernel.length) == 1:
                    key = (input_key, conn_key, 1)
                else:
                    key = (input_key, conn_key, 2, tuple(kernel.length.tolist()))

                if key in rep:
                    r = rep[key]

                    if len(kernel.length) == 1:
                        # share (matches your original intention)
                        kernel.ord_nn(
                            ord=r.ord,
                            rev_ord=r.rev_ord,
                            NNarray=r.NNarray,
                            NN_rev=r.NN_rev,
                            NN_count=r.NN_count,
                            pointer=compute_pointer
                        )
                    else:
                        # copy (matches your original intention)
                        kernel.ord_nn(
                            ord=r.ord.copy(),
                            rev_ord=r.rev_ord.copy(),
                            NNarray=r.NNarray.copy(),
                            NN_rev=r.NN_rev.copy(),
                            NN_count=r.NN_count.copy(),
                            pointer=compute_pointer
                        )
                else:
                    # first kernel in this group: compute fresh
                    kernel.ord_nn(pointer=compute_pointer)
                    rep[key] = kernel
    
    # def update_ord_nn(self):
    #     """Update order and KNN in each GP node for Vecchia approximation
    #     """
    #     n_layer=len(self.all_layer)
    #     for l in range(n_layer):
    #         layer=self.all_layer[l]
    #         for k, kernel in enumerate(layer):
    #             if kernel.type == 'gp':
    #                 compute_pointer = False if kernel.imp_pointer_row is None else True
    #                 if k == 0:
    #                     kernel.ord_nn(pointer=compute_pointer)
    #                 else:
    #                     if len(kernel.length) == 1:
    #                         found_match = False
    #                         for j in range(k):
    #                             if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
    #                                 kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
    #                                 found_match = True
    #                                 break
    #                         if not found_match:
    #                             kernel.ord_nn(pointer=compute_pointer)
    #                     else:
    #                         found_match = False
    #                         for j in range(k):
    #                             if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
    #                                 kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
    #                                 found_match = True
    #                                 break
    #                         if not found_match:
    #                             kernel.ord_nn(pointer=compute_pointer)
