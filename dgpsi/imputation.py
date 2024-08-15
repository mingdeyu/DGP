from numpy.random import uniform
import numpy as np
from .functions import update_f, fmvn
from .vecchia import fmvn_sp, U_matrix_sp, U_matrix_sp_rep

class imputer:
    """Class to implement imputation of latent variables.

    Args:
        all_layer (list): a list that contains the DGP model
        block (bool, optional): whether to use the blocked (layer-wise) ESS for the imputations. Defaults to `True`.
    """
    def __init__(self, all_layer, block=True):
        self.all_layer=all_layer
        self.block=block

    def __setstate__(self, state):
        if 'block' not in state:
            state['block'] = True
        self.__dict__.update(state)

    def sample(self,burnin=0):
        """Implement the imputation via the ESS-within-Gibbs.

        Args:
            burnin (int, optional): the number of burnin iterations for the ESS-within-Gibbs sampler
                to generate one realisation of latent variables. Defaults to `0`.
        """
        n_layer=len(self.all_layer)
        for _ in range(burnin+1):
            for l in range(n_layer-1):
                layer=self.all_layer[l]
                linked_layer=self.all_layer[l+1]
                is_hetero_type = np.any([True if kernel.type=='likelihood' and kernel.exact_post_idx!=None else False for kernel in linked_layer])
                if self.block and not is_hetero_type:
                    self.one_sample_block(layer,linked_layer)
                else:
                    n_kernel=len(layer)
                    for k in range(n_kernel):
                        target_kernel=layer[k]
                        linked_upper_kernels=[kernel for kernel in linked_layer if k in kernel.input_dim]
                        self.one_sample(target_kernel,linked_upper_kernels,k)

    @staticmethod
    def one_sample_block(target_layer,upper_layer):
        """Impute a latent layer.

        Args:
            target_layer (list): a list of GPs that produce a latent layer that needs to be imputed.
            upper_layer (list): a list of GPs (in the next layer) that are fed by the output of GPs in **target_layer**.
        """
        M, N = len(target_layer), len(target_layer[0].output)
        f, nu = np.zeros((N,M)), np.zeros((N,M))
        for i, kernel in enumerate(target_layer):
            f[:,i] = kernel.output.flatten()
            if kernel.vecch:
                if kernel.global_input is not None:
                    X=np.concatenate((kernel.input, kernel.global_input),1)
                else:
                    X=kernel.input
                nu[:,i] = fmvn_sp(X[kernel.ord], kernel.NNarray, kernel.scale[0], kernel.length, kernel.nugget[0], kernel.name)[kernel.rev_ord]
            else:
                nu[:,i] = fmvn(kernel.scale*kernel.k_matrix())

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
        else:
            covariance=target_kernel.k_matrix()
            covariance=target_kernel.scale*covariance
                  
        if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
            idx=np.where(linked_upper_kernels[0].input_dim == k)[0]
            if idx in linked_upper_kernels[0].exact_post_idx:
                if target_kernel.vecch:
                    if linked_upper_kernels[0].rep is not None:                     
                        Gamma = np.exp(linked_upper_kernels[0].input[:,1])
                        U_sp_latent, U_sp_obs_latent= U_matrix_sp_rep(X[target_kernel.ord], target_kernel.imp_NNarray, target_kernel.rep_hetero, target_kernel.ord, target_kernel.scale[0], target_kernel.length, target_kernel.nugget[0], target_kernel.name, Gamma, target_kernel.imp_pointer_row, target_kernel.imp_pointer_col)
                        f = linked_upper_kernels[0].posterior_vecch(idx=idx, U_sp_l=U_sp_latent, U_sp_ol=U_sp_obs_latent, ord=target_kernel.ord, rev_ord=target_kernel.rev_ord)    
                    else:
                        Gamma = np.exp(linked_upper_kernels[0].input[:,1])[target_kernel.ord]
                        U_sp_latent, U_sp_obs_latent = U_matrix_sp(X[target_kernel.ord], target_kernel.imp_NNarray, target_kernel.scale[0], target_kernel.length, target_kernel.nugget[0], target_kernel.name, np.concatenate((Gamma, Gamma)), target_kernel.imp_pointer_row, target_kernel.imp_pointer_col)
                        f=linked_upper_kernels[0].posterior_vecch(idx=idx, U_sp_l=U_sp_latent, U_sp_ol=U_sp_obs_latent, ord=target_kernel.ord, rev_ord=target_kernel.rev_ord)
                else:
                    f=linked_upper_kernels[0].posterior(idx=idx,v=covariance)
                if linked_upper_kernels[0].rep is None:
                    linked_upper_kernels[0].input[:,idx]=f.reshape(-1,1)
                else:
                    linked_upper_kernels[0].input[:,idx]=f[linked_upper_kernels[0].rep].reshape(-1,1)
                target_kernel.output[:,0]=f
                return
        
        f=(target_kernel.output).flatten()
        # Choose the ellipse for this sampling iteration.
        #nu = np.random.default_rng().multivariate_normal(mean=np.zeros(len(f)),cov=covariance,check_valid='ignore')  
        if target_kernel.vecch:
            nu = fmvn_sp(X[target_kernel.ord], target_kernel.NNarray, target_kernel.scale[0], target_kernel.length, target_kernel.nugget[0], target_kernel.name)[target_kernel.rev_ord]
        else:
            nu = fmvn(covariance)                       
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
                if linked_kernel.rep is None:
                    linked_kernel.input[:,linked_kernel.input_dim==k]=fp.reshape(-1,1)
                else:
                    linked_kernel.input[:,linked_kernel.input_dim==k]=fp[linked_kernel.rep].reshape(-1,1)
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
        """Update order and KNN in each GP node for Vecchia approximation
        """
        n_layer=len(self.all_layer)
        for l in range(n_layer):
            layer=self.all_layer[l]
            for k, kernel in enumerate(layer):
                if kernel.type == 'gp':
                    compute_pointer = False if kernel.imp_pointer_row is None else True
                    if k == 0:
                        kernel.ord_nn(pointer=compute_pointer)
                    else:
                        if len(kernel.length) == 1:
                            found_match = False
                            for j in range(k):
                                if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                    kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                    found_match = True
                                    break
                            if not found_match:
                                kernel.ord_nn(pointer=compute_pointer)
                        else:
                            found_match = False
                            for j in range(k):
                                if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                    kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                    found_match = True
                                    break
                            if not found_match:
                                kernel.ord_nn(pointer=compute_pointer)
