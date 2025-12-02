import numpy as np
from scipy.special import gammaln, expit, log_ndtr, ndtr, owens_t
from scipy.linalg import cholesky, cho_solve
#from scipy.sparse.linalg import spsolve_triangular
#from .functions import categorical_sampler #fmvn_mu
from .vecchia import forward_substitute, add_to_diag_square

class Poisson:
    """Class to implement Poisson likelihood. It can only be added as the final layer of a DGP model.

    Args:
        input_dim (ndarray, optional): a numpy 1d-array of length one that contains the indices of one GP in the feeding 
            layer whose outputs feed into the likelihood node. When set to `None`, all outputs from GPs of 
            the feeding layer feed into the likelihood node, and in this case one needs to ensure there is only one GP node specified
            in the feeding layer. Defaults to `None`.

    Attributes:
        type (str): identifies that the node is a likelihood node;
        input (ndarray): a numpy 2d-array (each row as a data point and each column as a likelihood parameter from the
            DGP part) that contains the input data (according to the argument **input_dim**) to the likelihood node. The value of 
            this attribute is assigned during the initialisation of :class:`.dgp` class. 
        output (ndarray): a numpy 2d-array with only one column that contains the output data to the likelihood node.
            The value of this attribute is assigned during the initialisation of :class:`.dgp` class.
        exact_post_idx (ndarray): a numpy 1d-array that indicates the indices of the likelihood parameters that allow closed-form
            conditional posterior distributions. Defaults to `None`.
        rep (ndarray): a numpy 1d-array used to re-construct repetitions in the data according to the repetitions in the global input,
            i.e., rep is assigned during the initialisation of :class:`.dgp` class if one input position has multiple outputs. Otherwise, it is
            `None`. Defaults to `None`. 
    """
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='Poisson'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.exact_post_idx=None
        self.rep=None

    def llik(self):
        """The log-likelihood function of Poisson distribution.

        Returns:
            ndarray: a numpy 1d-array of log-likelihood.
        """
        llik=self.output*self.input-np.exp(self.input)-gammaln(self.output+1)
        llik=np.sum(llik)
        return llik
    
    @staticmethod
    def pllik(y,f):
        """The predicted log-likelihood function of Poisson distribution.

        Args:
            y (ndarray): a numpy 3d-array of output data with shape ``(N,1,1)``, where *N* is the number of output data points.
            f (ndarray): a numpy 3d-array of sample points with shape ``(N,S,Q)``, where *S* is the number of sample points and 
                *Q* is the number of parameters in the distribution (e.g., *Q* = `1` for Poisson distribution).

        Returns:
            ndarray: a numpy 3d-array of log-likelihood for given **f**.
        """
        pllik=y*f-np.exp(f)-gammaln(y+1)
        return pllik

    @staticmethod    
    def prediction(m,v):
        """Compute mean and variance of the DGP+Poisson model given the predictive mean and variance of DGP model for Poisson parameter.
        
        Args:
            m (ndarray): a numpy 2d-array of predictive mean from the DGP model for the Poisson parameter.
            v (ndarray): a numpy 2d-array of predictive variance from the DGP model for the Poisson parameter.

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions (that are 
            represented by predictive means and variances).
        """
        y_mean=np.exp(m+v/2)
        y_var=np.exp(m+v/2)+(np.exp(v)-1)*np.exp(2*m+v)
        return y_mean.flatten(),y_var.flatten()
    
    def sampling(self,f_sample):
        """Generate samples of DGP+Poisson model given samples of DGP model for the Poisson parameter.
        
        Args:
            f_sample (ndarray): a numpy 2d-array (with one column) of samples from the DGP model for the Poisson parameter.
           
        Returns:
            tuple: a tuple of one 1d-arrays giving samples at the testing input data positions.
        """
        y_sample=np.random.poisson(np.exp(f_sample))
        return y_sample.flatten()

class Hetero:
    """Class to implement Heteroskedastic Gaussian likelihood. It can only be added as the final layer of a DGP model.

    Args:
        input_dim (ndarray, optional): a numpy 1d-array of length two that contains the indices of two GPs in the feeding 
            layer whose outputs feed into the likelihood node. When set to `None`, all outputs from GPs of 
            the feeding layer feed into the likelihood node, and in this case one needs to ensure there are only two GP nodes specified
            in the feeding layer. Defaults to `None`.
    """
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='Hetero'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.exact_post_idx=np.array([0])
        self.rep=None

    def llik(self):
        mu,log_var=self.input[:,0],self.input[:,1]
        r2 = ((self.output).flatten()-mu)**2
        llik=-0.5*(np.log(2*np.pi)+log_var+np.exp(np.log(r2)-log_var))
        llik=np.sum(llik)
        return llik

    @staticmethod
    def pllik(y,f):
        mu,var=f[:,:,[0]],np.exp(f[:,:,[1]])
        pllik=-0.5*(np.log(2*np.pi*var)+(y-mu)**2/var)
        return pllik

    @staticmethod    
    def prediction(m,v):
        y_mean=m[:,0]
        y_var=np.exp(m[:,1]+v[:,1]/2)+v[:,0]
        return y_mean.flatten(),y_var.flatten()
    
    @staticmethod
    def sampling(f_sample):
        y_sample=np.random.normal(f_sample[:,0],np.sqrt(np.exp(f_sample[:,1])))
        return y_sample.flatten()

    def posterior(self,idx,v):
        """Sampling from the conditional posterior distribution of the mean in heteroskedastic Gaussian likelihood.
        """
        if idx==0:
            if self.rep is None:
                Gamma=np.exp(self.input[:,1])
                #y=(self.output).flatten()
                f_mu=self.post_het1(v,Gamma,self.output)
            else:
                Gamma=np.exp(self.input[:,1])
                #y_mask=self.output[:,0]
                #mask_f=self.rep
                #v_mask=v[mask_f,:]
                #V_mask=v[mask_f,:][:,mask_f]
                #mu,cov=self.post_het2(v,Gamma,v_mask,V_mask,y_mask)
                f_mu=self.post_het2(v,Gamma,self.rep,self.output)
            #f_mu=fmvn_mu(mu,cov)
            return f_mu
        
    def posterior_vecch(self, idx, U_sp_l, U_sp_ol, ord, rev_ord, invd = None, invg = None):
        """Sampling from the conditional posterior distribution of the mean in heteroskedastic Gaussian likelihood under the Vecchia Approximation.
        """
        if idx==0:
            if self.rep is None:
                f_mu = self.post_het_vecch(U_sp_l, U_sp_ol, self.output[ord,0])[rev_ord]
            else:
                num = np.bincount(self.rep, weights=invg*self.output.flatten(), minlength=U_sp_l.shape[0])[ord]
                y = num * invd
                f_mu = self.post_het_vecch(U_sp_l, U_sp_ol, y)[rev_ord]
            return f_mu
        
    @staticmethod
    def post_het_vecch(U_sp_l, U_sp_ol, y):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are repetitions
           in the training data under the Vecchia approximation.
        """
        L_sp_l = U_sp_l.transpose().tocsr()
        intermediate = U_sp_ol.transpose().dot(y)
        #U_latent_obs_y = U_sp_ol.transpose().dot(y)
        #U_latent_U_latent_obs_y = U_sp_l.dot(U_latent_obs_y)
        #intermediate = backward_substitute(U_sp_l.data, U_sp_l.indices, U_sp_l.indptr, U_latent_obs_y)
        mu = -forward_substitute(L_sp_l.data, L_sp_l.indices, L_sp_l.indptr, intermediate)
        #mu = -spsolve_triangular(L_sp_l, intermediate)
        sd = np.random.randn(U_sp_l.shape[0])
        samp = forward_substitute(L_sp_l.data, L_sp_l.indices, L_sp_l.indptr, sd)
        #samp = spsolve_triangular(L_sp_l, sd)
        f = mu + samp
        return f

    @staticmethod
    def post_het1(v,Gamma,y_mask):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are no repetitions
           in the training data.
        """
        #N = v.shape[0]
        vGamma = v.copy()
        #add_to_diag_square(vGamma, np.full(N, 1e-10))
        #v_jitter = vGamma.copy()
        add_to_diag_square(vGamma, Gamma)
    
        L = cholesky(vGamma, lower=True, check_finite=False)

        L1 = cholesky(v, lower=True, check_finite=False)
        mu = v.dot(cho_solve((L, True), y_mask.flatten(), check_finite=False))
        sd = np.random.randn(len(mu),2)
        u = L1.dot(sd[:,0])
        w = np.sqrt(Gamma) * sd[:,1]
        f = -v.dot(cho_solve((L, True), u+w, check_finite=False))
        f += (mu + u)

        # Linvv = solve_triangular(L, v, lower=True, trans=0, check_finite=False)
        # cov = v - Linvv.T@Linvv
        # mu = cov.dot(y_mask.flatten()/Gamma)
        return f

    @staticmethod
    def post_het2(v,Gamma,mask_f,y_mask):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are repetitions
           in the training data.
        """
        N = v.shape[0]
        GammaInv = 1.0/Gamma
        GammaInvY = GammaInv * y_mask.flatten()
        MGammaInvY = np.bincount(mask_f, weights=GammaInvY, minlength=N)
        MGammaInvM = np.bincount(mask_f, weights=GammaInv, minlength=N)

        invMGammaInvM = 1.0/MGammaInvM
        vinvMGammaInvM = v.copy()
        #add_to_diag_square(vinvMGammaInvM, np.full(N, 1e-10))
        #v_jitter = vinvMGammaInvM.copy()
        add_to_diag_square(vinvMGammaInvM, invMGammaInvM)

        L = cholesky(vinvMGammaInvM, lower=True, check_finite=False)

        # Linvv = solve_triangular(L, v, lower=True, trans=0, check_finite=False)
        # cov = v - Linvv.T@Linvv
        # mu = cov.dot(MGammaInvY)

        L1 = cholesky(v, lower=True, check_finite=False)
        mu=v.dot(cho_solve((L, True), invMGammaInvM*MGammaInvY, check_finite=False))

        sd = np.random.randn(len(mu),2)
        u = L1.dot(sd[:,0])
        w = np.sqrt(invMGammaInvM) * sd[:,1]
        f = -v.dot(cho_solve((L, True), u+w, check_finite=False))
        f += (mu + u)
        return f    

class NegBin:
    """Class to implement Negative Binomial likelihood. It can only be added as the final layer of a DGP model.

    Args:
        input_dim (ndarray, optional): a numpy 1d-array of length two that contains the indices of two GPs in the feeding 
            layer whose outputs feed into the likelihood node. When set to `None`, all outputs from GPs of 
            the feeding layer feed into the likelihood node, and in this case one needs to ensure there are only two GP nodes specified
            in the feeding layer. Defaults to `None`.
    """
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='NegBin'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.exact_post_idx=None
        self.rep=None
        #self.rep_sp=None
    
    def llik(self):
        y,f1,f2=(self.output).flatten(),self.input[:,0],self.input[:,1]
        n = np.exp(-f2)
        a = f1 + f2
        softplus_a = np.logaddexp(0.0, a)
        llik=gammaln(y+n)-gammaln(n)-gammaln(y+1.0)+y*a-(y+n)*softplus_a
        llik=np.sum(llik)
        return llik

    @staticmethod
    def pllik(y,f):
        f1, f2 = f[:,:,[0]], f[:,:,[1]]
        n = np.exp(-f2)
        a = f1 + f2
        softplus_a = np.logaddexp(0.0, a)
        pllik=gammaln(y+n)-gammaln(n)-gammaln(y+1.0)+y*a-(y+n)*softplus_a
        return pllik
    
    @staticmethod
    def prediction(m,v):
        y_mean=np.exp(m[:,0]+v[:,0]/2)
        y_var=np.exp(2*m[:,0]+v[:,0])*(np.exp(v[:,0])-1)+np.exp(m[:,0]+v[:,0]/2)+np.exp(m[:,1]+v[:,1]/2)*np.exp(2*m[:,0]+2*v[:,0])
        return y_mean.flatten(),y_var.flatten()
    
    @staticmethod
    def sampling(f_sample):
        p, k=1/(1+np.exp(f_sample[:,0]+f_sample[:,1])), np.exp(-f_sample[:,1])
        y_sample=np.random.negative_binomial(k,p)
        return y_sample.flatten()

class Categorical:
    """Class to implement categorical likelihood for binary and multi-class classifications. It can only be added as the final layer of a DGP model.

    Args:
        num_classes (int, optional): an integer indicating the number of classes in the training data.
        input_dim (ndarray, optional): a numpy 1d-array of length one that contains the indices of one GP (if the output has two classes) and K
            (if the output has K > 2 classes) in the feeding layer whose outputs feed into the likelihood node. When set to `None`, 
            all outputs from GPs of the feeding layer feed into the likelihood node, and in this case one needs to ensure there is only one GP
            node (for binary classification) or K GP nodes (for multi-class classification) specified in the feeding layer. Defaults to `None`.
        link (str, optional): the link function to be used for binary classification. Either 'probit' or 'logit' for binary classification, or
            'robustmax' or 'softmax' for multi-class classification. Defaults to 'None'.
        robustmax_eps (float, optional): Noise / smoothing parameter for the robustmax link function.

    Attributes:
        type (str): identifies that the node is a likelihood node;
        input (ndarray): a numpy 2d-array (each row as a data point and each column as a likelihood parameter from the
            DGP part) that contains the input data (according to the argument **input_dim**) to the likelihood node. The value of 
            this attribute is assigned during the initialisation of :class:`.dgp` class. 
        output (ndarray): a numpy 2d-array with only one column that contains the output data to the likelihood node.
            The value of this attribute is assigned during the initialisation of :class:`.dgp` class.
        exact_post_idx (ndarray): a numpy 1d-array that indicates the indices of the likelihood parameters that allow closed-form
            conditional posterior distributions. Defaults to `None`.
        rep (ndarray): a numpy 1d-array used to re-construct repetitions in the data according to the repetitions in the global input,
            i.e., rep is assigned during the initialisation of :class:`.dgp` class if one input position has multiple outputs. Otherwise, it is
            `None`. Defaults to `None`. 
    """
    def __init__(self, num_classes=None, input_dim=None, link = None, robustmax_eps=1e-3):
        self.type='likelihood'
        self.name='Categorical'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.exact_post_idx=None
        self.rep=None
        self.num_classes=num_classes
        self.class_encoder=None
        self.link=link
        self.robustmax_eps = robustmax_eps

    def llik(self):
        """The log-likelihood function of Categorical distribution.

        Returns:
            ndarray: a numpy 1d-array of log-likelihood.
        """
        if self.num_classes==2:
            if self.link=='logit':
                llik = np.sum(self.output * self.input - np.logaddexp(0, self.input))
            else:
                llik = np.sum(self.output * log_ndtr(self.input) + (1-self.output) * log_ndtr(-self.input))
        else: 
            if self.link == 'robustmax': 
                K = self.num_classes
                eps = self.robustmax_eps
                k_star = np.argmax(self.input, axis=1)
                y = self.output.flatten().astype(int)
                correct = (y == k_star)
                p_correct = 1.0 - eps
                p_wrong = eps / (K - 1)
                llik = np.sum(np.where(correct, np.log(p_correct), np.log(p_wrong)))
            else:
                max_logits = np.max(self.input, axis=1, keepdims=True)
                stable_exp = np.exp(self.input - max_logits)
                log_sum_exp = np.log(np.sum(stable_exp, axis=1)) + max_logits.flatten()
                llik = np.sum(self.input[np.arange(len(self.output)), self.output.flatten()] - log_sum_exp)
        return llik
    
    def pllik(self, y, f):
        if self.num_classes==2:
            if self.link=='logit':
                pllik = y * f - np.logaddexp(0, f)
            else:
                pllik = y * log_ndtr(f) + (1-y) * log_ndtr(-f)
        else:
            if self.link == 'robustmax': 
                K = self.num_classes
                eps = self.robustmax_eps
                k_star = np.argmax(f, axis=2)
                y_flat = y.flatten().astype(int)
                correct = (k_star == y_flat[:, None])
                p_correct = 1.0 - eps
                p_wrong = eps / (K - 1)
                pllik = np.where(correct, np.log(p_correct), np.log(p_wrong))[:, :, None]
            else:
                max_logits = np.max(f, axis=2, keepdims=True)
                stable_exp = np.exp(f - max_logits)
                log_sum_exp = np.log(np.sum(stable_exp, axis=2)) + np.squeeze(max_logits, axis=2)
                pllik = (f[np.arange(len(y)), :, y.flatten()] - log_sum_exp)[:, :, None]
        return pllik
    
    def prediction(self, m, v):
        if self.num_classes==2:
            if self.link=='logit':
                m, v = m.flatten(), v.flatten()
                denom = 1.0 + (np.pi/8.0) * v
                mu_star = m / np.sqrt(denom)
                y_mean = expit(mu_star)
                var_star = v / denom
                y_var = (y_mean * (1.0 - y_mean))**2 * var_star
                y_var = np.clip(y_var, 0.0, y_mean * (1.0 - y_mean))
                y_mean, y_var = y_mean.reshape(-1,1), y_var.reshape(-1,1)
            else:
                m, v = m.flatten(), v.flatten()
                t = m / np.sqrt(1.0 + v)        # m / sqrt(1+v)
                y_mean = ndtr(t)                # Φ(t)
                a = 1.0 / np.sqrt(1.0 + 2.0*v)  # 1 / sqrt(1+2v)
                Ep2 = y_mean - 2.0 * owens_t(t, a)  # Φ(t) - 2*T(t,a) = E[p^2]
                y_var = Ep2 - y_mean*y_mean
                # Guard tiny negative numerical noise:
                y_var = np.maximum(y_var, 0.0)
                y_mean, y_var = y_mean.reshape(-1,1), y_var.reshape(-1,1)
        else:
            if self.link == 'robustmax':
                K = self.num_classes
                eps = self.robustmax_eps
                S = 1000
                chunk_size = 200                                   
                std = np.sqrt(np.maximum(v, 0.0))
                win_counts = np.zeros((m.shape[0], K), dtype=float)  
                done = 0                                            
                while done < S:                                      
                    this = min(chunk_size, S - done)                 
                    f_chunk = m[:, None, :] + std[:, None, :] * np.random.randn(m.shape[0], this, K) 
                    k_star = np.argmax(f_chunk, axis=2)              
                    np.add.at(win_counts, (np.arange(m.shape[0])[:, None], k_star), 1.0)  
                    done += this                                    
                q_hat = win_counts / S                              
                a = 1.0 - eps                                        
                b = eps / (K - 1)                                    
                y_mean = b + (a - b) * q_hat                      
                y_var  = (a - b)**2 * q_hat * (1.0 - q_hat)
            else:
                K = self.num_classes
                S = 1000
                if S % 2 == 1:
                    S += 1                                               
                chunk_size = 200                                         
                std = np.sqrt(np.maximum(v, 0.0))
                sum_p  = np.zeros((m.shape[0], K), dtype=float)          
                sum_p2 = np.zeros((m.shape[0], K), dtype=float)         
                done = 0                                              
                while done < S:                                         
                    this = min(chunk_size, S - done)                 
                    half = (this + 1) // 2                             
                    eps_half = np.random.randn(m.shape[0], half, K)     
                    eps = np.concatenate([eps_half, -eps_half], axis=1)[:, :this, :] 
                    f_samp = m[:, None, :] + std[:, None, :] * eps     
                    f_samp -= np.max(f_samp, axis=2, keepdims=True)    
                    np.exp(f_samp, out=f_samp)                        
                    f_samp /= np.sum(f_samp, axis=2, keepdims=True)    
                    sum_p  += f_samp.sum(axis=1)                        
                    sum_p2 += (f_samp * f_samp).sum(axis=1)             
                    done += this                                       
                y_mean = sum_p / S                                      
                y_var  = sum_p2 / S - y_mean**2    
        return y_mean,y_var
    
    def sampling(self, f_sample):
        if self.num_classes==2:
            if self.link=='logit':
                y_sample = expit(f_sample)
            else:
                y_sample = ndtr(f_sample)
            #y_sample = np.concatenate((1-prob_sample, prob_sample), axis=1)
        else:
            if self.link == 'robustmax':
                K = self.num_classes
                eps = self.robustmax_eps
                k_star = np.argmax(f_sample, axis=1)
                y_sample = np.full_like(f_sample, eps/(K-1), dtype=float)
                y_sample[np.arange(f_sample.shape[0]), k_star] = 1.0 - eps
            else:
                exp_logit = np.exp(f_sample - np.max(f_sample, axis=1, keepdims=True))
                y_sample = exp_logit/np.sum(exp_logit, axis=1, keepdims=True)
        return y_sample
    
class ZIP:
    """Class to implement Zero-Inflated Poisson (ZIP) likelihood. It can only be added
    as the final layer of a DGP model.

    This likelihood assumes two latent GP outputs at each input location:
    one controlling the Poisson mean (on the log scale) and one controlling
    the zero-inflation probability (on the logit scale).

    Args:
        input_dim (ndarray, optional): a numpy 1d-array of length two that contains
            the indices of two GPs in the feeding layer whose outputs feed into the
            likelihood node. The first index is interpreted as the GP for the log
            Poisson mean, and the second as the GP for the logit zero-inflation
            probability. When set to `None`, all outputs from GPs of the feeding
            layer feed into the likelihood node, and in this case one needs to
            ensure there are only two GP nodes specified in the feeding layer.
            Defaults to `None`.
    """
    def __init__(self, input_dim=None):
        self.type = 'likelihood'
        self.name = 'ZIP'
        self.input = None    
        self.output = None 
        self.input_dim = input_dim
        self.exact_post_idx = None
        self.rep = None

    def llik(self):
        """
        Log-likelihood of Zero-Inflated Poisson.
        """
        y = (self.output).flatten()
        f_lambda = self.input[:, 0]
        f_pi = self.input[:, 1]

        lam = np.exp(f_lambda)
        pi  = expit(f_pi)

        mask0   = (y == 0)
        maskpos = ~mask0

        ll = np.empty_like(y, dtype=float)

        log_pi      = np.log(pi[mask0])
        log1m_pi_0  = np.log1p(-pi[mask0])
        ll[mask0]   = np.logaddexp(log_pi, log1m_pi_0 - lam[mask0])

        log1m_pi_pos = np.log1p(-pi[maskpos])
        ll[maskpos] = (
            log1m_pi_pos
            - lam[maskpos]
            + y[maskpos] * f_lambda[maskpos]
            - gammaln(y[maskpos] + 1.0)
        )
        return np.sum(ll)

    @staticmethod
    def pllik(y, f):
        """
        Predicted log-likelihood for Zero-Inflated Poisson.

        Args:
            y (ndarray): shape (N, 1, 1), observed counts.
            f (ndarray): shape (N, S, 2),

        Returns:
            ndarray: log-likelihoods with shape (N, S, 1).
        """
        eta_lambda = f[..., 0]     # (N, S)
        eta_pi     = f[..., 1]     # (N, S)

        f_lambda = eta_lambda[..., None]      # (N, S, 1)
        lam      = np.exp(f_lambda)           # (N, S, 1)

        pi = expit(eta_pi)[..., None]         # (N, S, 1)

        # Broadcast y and log(y!) to match (N, S, 1)
        y_b   = np.broadcast_to(y, lam.shape)                  # (N, S, 1)
        log_y = gammaln(y + 1.0)                               # (N, 1, 1)
        log_y_b = np.broadcast_to(log_y, lam.shape)            # (N, S, 1)

        ll = np.empty_like(lam, dtype=float)   # (N, S, 1)

        # Masks for zeros and positives
        mask0   = (y_b == 0)
        maskpos = ~mask0

        pi0   = pi[mask0]
        lam0  = lam[mask0]
        log_pi0     = np.log(pi0)
        log1m_pi0   = np.log1p(-pi0)
        ll[mask0]   = np.logaddexp(log_pi0, log1m_pi0 - lam0)

        log1m_pi_pos = np.log1p(-pi[maskpos])
        ll[maskpos] = (
            log1m_pi_pos
            - lam[maskpos]
            + y_b[maskpos] * f_lambda[maskpos]
            - log_y_b[maskpos]
        )

        return ll
    
    @staticmethod    
    def prediction(m, v):
        """Compute mean and variance 
        """
        m_lambda = m[:, 0]
        v_lambda = v[:, 0]
        m_pi     = m[:, 1]
        v_pi     = v[:, 1]

        # Moments of lambda
        lam_mean = np.exp(m_lambda + 0.5 * v_lambda)
        lam_var  = (np.exp(v_lambda) - 1.0) * np.exp(2.0 * m_lambda + v_lambda)

        # Moments of pi
        denom   = 1.0 + (np.pi / 8.0) * v_pi
        denom   = np.maximum(denom, 1e-12)  # numerical safety
        mu_star = m_pi / np.sqrt(denom)

        pi_mean = expit(mu_star) 
        var_star = v_pi / denom
        pi_var   = (pi_mean * (1.0 - pi_mean))**2 * var_star
        pi_var   = np.clip(pi_var, 0.0, pi_mean * (1.0 - pi_mean))

        # Predictive mean
        y_mean = (1.0 - pi_mean) * lam_mean

        # Predictive variance
        cond_var = (1.0 - pi_mean) * lam_mean * (1.0 + pi_mean * lam_mean)
        var_g = ((1.0 - pi_mean)**2 + pi_var) * lam_var + pi_var * lam_mean**2
        y_var = cond_var + var_g
        y_var = np.maximum(y_var, 0.0)  # numerical safety

        return y_mean.flatten(), y_var.flatten()
    
    def sampling(self, f_sample):
        """Generate samples
        """
        eta_lambda = f_sample[:, 0]
        eta_pi     = f_sample[:, 1]

        lam = np.exp(eta_lambda)
        pi  = expit(eta_pi)

        N = f_sample.shape[0]
        u = np.random.rand(N)

        # Structural zero with prob pi, otherwise Poisson
        y = np.where(u < pi, 0, np.random.poisson(lam))

        return y.flatten()

class ZINB:
    """Class to implement Zero-Inflated Negative Binomial (ZINB) likelihood.
    It can only be added as the final layer of a DGP model.

    This likelihood assumes three latent GP outputs at each input location:
    one controlling the Negative Binomial mean on the log scale, one
    controlling the dispersion / size parameter, and one controlling the
    zero-inflation probability on the logit scale.

    Args:
        input_dim (ndarray, optional): a numpy 1d-array of length three that
            contains the indices of three GPs in the feeding layer whose
            outputs feed into the likelihood node. The first index is
            interpreted as the GP for the log-mean (log mu), the second as
            the GP for the dispersion (-log n), and the third as the GP for
            the logit zero-inflation probability. When set to `None`, all
            outputs from GPs of the feeding layer feed into the likelihood
            node, and in this case one needs to ensure there are exactly
            three GP nodes specified in the feeding layer. Defaults to `None`.
    """
    def __init__(self, input_dim=None):
        self.type = 'likelihood'
        self.name = 'ZINB'
        self.input = None
        self.output = None
        self.input_dim = input_dim
        self.exact_post_idx = None
        self.rep = None

    def llik(self):
        """
        Log-likelihood of Zero-Inflated Negative Binomial.
        """
        y = (self.output).flatten()
        f1 = self.input[:, 0]  
        f2 = self.input[:, 1] 
        f_pi = self.input[:, 2] 

        # NB parameters
        n = np.exp(-f2)
        a = f1 + f2    
        softplus_a = np.logaddexp(0.0, a)

        # log NB pmf for all y
        log_nb = (
            gammaln(y + n)
            - gammaln(n)
            - gammaln(y + 1.0)
            + y * a
            - (y + n) * softplus_a
        )

        pi = expit(f_pi)

        mask0 = (y == 0)
        maskpos = ~mask0

        ll = np.empty_like(y, dtype=float)

        log_nb0 = log_nb[mask0]
        log_pi0 = np.log(pi[mask0])
        log1m_pi0 = np.log1p(-pi[mask0])

        ll[mask0] = np.logaddexp(log_pi0, log1m_pi0 + log_nb0)

        log1m_pi_pos = np.log1p(-pi[maskpos])
        ll[maskpos] = log1m_pi_pos + log_nb[maskpos]

        return np.sum(ll)

    @staticmethod
    def pllik(y, f):
        """
        Predicted log-likelihood for Zero-Inflated Negative Binomial.
        """
        # Ensure arrays
        y = np.asarray(y)           # (N,1,1)
        f = np.asarray(f)           # (N,S,3)

        f1 = f[..., 0:1]            # (N,S,1)
        f2 = f[..., 1:2]            # (N,S,1)
        f_pi = f[..., 2:3]          # (N,S,1)

        n = np.exp(-f2)             # (N,S,1)
        a = f1 + f2                 # (N,S,1)
        softplus_a = np.logaddexp(0.0, a)

        # Broadcast y to (N,S,1)
        y_b = np.broadcast_to(y, n.shape)

        # log NB pmf for all (N,S,1)
        log_nb = (
            gammaln(y_b + n)
            - gammaln(n)
            - gammaln(y_b + 1.0)
            + y_b * a
            - (y_b + n) * softplus_a
        )

        pi = expit(f_pi)            # (N,S,1)

        ll = np.empty_like(log_nb, dtype=float)

        mask0 = (y_b == 0)
        maskpos = ~mask0

        log_nb0 = log_nb[mask0]
        log_pi0 = np.log(pi[mask0])
        log1m_pi0 = np.log1p(-pi[mask0])
        ll[mask0] = np.logaddexp(log_pi0, log1m_pi0 + log_nb0)

        log1m_pi_pos = np.log1p(-pi[maskpos])
        ll[maskpos] = log1m_pi_pos + log_nb[maskpos]

        return ll
    
    @staticmethod
    def prediction(m, v):
        """Compute predictive mean and variance of the ZINB model given the
        predictive mean and variance of the latent GP outputs.
        """
        m = np.asarray(m)
        v = np.asarray(v)

        m1   = m[:, 0]
        v1   = v[:, 0]
        m2   = m[:, 1]
        v2   = v[:, 1]
        m_pi = m[:, 2]
        v_pi = v[:, 2]

        # Moments of mu
        mu_mean = np.exp(m1 + 0.5 * v1)
        mu_var  = (np.exp(v1) - 1.0) * np.exp(2.0 * m1 + v1)
        mu2_mean = np.exp(2.0 * m1 + 2.0 * v1)

        E_exp_f2 = np.exp(m2 + 0.5 * v2)
        mu2_over_n_mean = mu2_mean * E_exp_f2 

        # Moments of pi
        denom = 1.0 + (np.pi / 8.0) * v_pi
        denom = np.maximum(denom, 1e-12)
        mu_star = m_pi / np.sqrt(denom)

        pi_mean = expit(mu_star) 
        var_star = v_pi / denom
        pi_var = (pi_mean * (1.0 - pi_mean))**2 * var_star
        pi_var = np.clip(pi_var, 0.0, pi_mean * (1.0 - pi_mean))

        # Predictive mean
        y_mean = (1.0 - pi_mean) * mu_mean

        # Predictive variance
        E_pi1m = pi_mean * (1.0 - pi_mean) - pi_var
        E_pi1m = np.clip(E_pi1m, 0.0, pi_mean * (1.0 - pi_mean))

        cond_var = (
            (1.0 - pi_mean) * (mu_mean + mu2_over_n_mean)
            + E_pi1m * mu2_mean
        )

        var_g = ((1.0 - pi_mean)**2 + pi_var) * mu_var + pi_var * (mu_mean**2)

        y_var = cond_var + var_g
        y_var = np.maximum(y_var, 0.0)

        return y_mean.flatten(), y_var.flatten()
    
    @staticmethod
    def sampling(f_sample):
        """Generate samples from the ZINB model given samples of the latent
        GP outputs.
        """
        f_sample = np.asarray(f_sample)
        f1 = f_sample[:, 0]
        f2 = f_sample[:, 1]
        f_pi = f_sample[:, 2]

        # NB parameters
        k = np.exp(-f2)
        p = 1.0 / (1.0 + np.exp(f1 + f2))

        # Zero-inflation
        pi = expit(f_pi)

        N = f_sample.shape[0]
        u = np.random.rand(N)

        nb_sample = np.random.negative_binomial(k, p)
        y = np.where(u < pi, 0, nb_sample)

        return y.flatten()