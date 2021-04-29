from numpy.random import uniform
import numpy as np
from .functions import log_likelihood_func, cmvn, fmvn, k_one_matrix, update_f

class imputer:
    """Class to implement imputation of latent variables.

    Args:
        all_layer (list): a list that contains the DGP model
    """
    def __init__(self, all_layer):
        self.all_layer=all_layer

    def sample(self,burnin=0):
        """Implement the imputation via the ESS-within-Gibbs.

        Args:
            burnin (int, optional): the number of burnin iterations for the ESS-within-Gibbs sampler
            to generate one realisation of latent variables. Defaults to 0.
        """
        n_layer=len(self.all_layer)
        for _ in range(burnin+1):
            for l in range(n_layer):
                layer=self.all_layer[l]
                n_kernel=len(layer)
                for k in range(n_kernel):
                    target_kernel=layer[k]
                    if l==n_layer-1:
                        if np.any(target_kernel.missingness):
                            mean, covariance = cmvn(target_kernel.input,target_kernel.global_input,target_kernel.output,target_kernel.scale,target_kernel.length,target_kernel.nugget,target_kernel.name,target_kernel.missingness)
                            target_kernel.output[target_kernel.missingness,0]=np.random.default_rng().multivariate_normal(mean=mean,cov=covariance,check_valid='ignore')
                    else:
                        linked_upper_kernels=[kernel for kernel in self.all_layer[l+1] if k in kernel.input_dim]
                        if np.any(target_kernel.missingness):
                            self.one_sample(target_kernel,linked_upper_kernels,k)
    
    @staticmethod
    def one_sample(target_kernel,linked_upper_kernels,k):
        """Impute one latent variable produced by a particular GP.

        Args:
            target_kernel (class): the GP whose output is a latent variable that needs to be imputed.
            linked_upper_kernels (list): a list of GPs (in the next layer) that link the output produced
                by the GP defined by the argument 'target_kernel'.
            k (int): the index indicating the position of the GP defined by the argument 'target_kernel' in
                its layer.
        """
        x=target_kernel.input
        f=target_kernel.output[target_kernel.missingness,0]
        if np.all(target_kernel.missingness):
            if np.any(target_kernel.connect!=None):
                x=np.concatenate((x, target_kernel.global_input),1)
            mean = np.zeros(len(f))
            covariance=k_one_matrix(x,target_kernel.length,target_kernel.name)+target_kernel.nugget*np.identity(len(x))
            covariance=target_kernel.scale*covariance
            # Choose the ellipse for this sampling iteration.
            nu = np.random.default_rng().multivariate_normal(mean=mean,cov=covariance,check_valid='ignore')
        else:
            mean, covariance = cmvn(x,target_kernel.global_input,target_kernel.output,target_kernel.scale,target_kernel.length,target_kernel.nugget,target_kernel.name,target_kernel.missingness)
            nu = np.random.default_rng().multivariate_normal(mean=mean,cov=covariance,check_valid='ignore')
        # Set the candidate acceptance threshold.
        log_y=0
        for linked_kernel in linked_upper_kernels:
            w=linked_kernel.input
            y=(linked_kernel.output).flatten()
            if np.any(linked_kernel.connect!=None):
                w=np.concatenate((w, linked_kernel.global_input),1)       
            cov_w=k_one_matrix(w,linked_kernel.length,linked_kernel.name)+linked_kernel.nugget*np.identity(len(w))
            log_y += log_likelihood_func(y,cov_w,linked_kernel.scale)
        log_y += np.log(uniform())
        # Set the bracket for selecting candidates on the ellipse.
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Iterates until an candidate is selected.
        while True:
            # Generates a point on the ellipse defines by `nu` and the input. We
            # also compute the log-likelihood of the candidate and compare to
            # our threshold.
            fp = update_f(f,nu,theta,mean)
            log_yp=0
            for linked_kernel in linked_upper_kernels:
                linked_kernel.input[target_kernel.missingness,linked_kernel.input_dim==k]=fp
                wp=linked_kernel.input
                y=(linked_kernel.output).flatten()
                if np.any(linked_kernel.connect!=None):
                    wp=np.concatenate((wp,linked_kernel.global_input),1)
                cov_wp=k_one_matrix(wp,linked_kernel.length,linked_kernel.name)+linked_kernel.nugget*np.identity(len(wp))
                log_yp += log_likelihood_func(y,cov_wp,linked_kernel.scale)
            if log_yp > log_y:
                target_kernel.output[target_kernel.missingness,0]=fp
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
