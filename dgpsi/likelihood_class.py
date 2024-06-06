import numpy as np
from scipy.special import loggamma
from scipy.linalg import cholesky, cho_solve
from scipy.sparse import diags_array
from .functions import fmvn_mu
from .vecchia import backward_substitute, forward_substitute

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
        #self.rep_sp=None

    def llik(self):
        """The log-likelihood function of Poisson distribution.

        Returns:
            ndarray: a numpy 1d-array of log-likelihood.
        """
        llik=self.output*self.input-np.exp(self.input)-loggamma(self.output+1)
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
        pllik=y*f-np.exp(f)-loggamma(y+1)
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
        #self.rep_sp=None

    def llik(self):
        mu,var=self.input[:,0],np.exp(self.input[:,1])
        llik=-0.5*(np.log(2*np.pi*var)+((self.output).flatten()-mu)**2/var)
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
                Gamma=np.diag(np.exp(self.input[:,1]))
                #y=(self.output).flatten()
                mu,cov=self.post_het1(v,Gamma,self.output)
            else:
                Gamma=np.exp(self.input[:,1])
                #y_mask=self.output[:,0]
                #mask_f=self.rep
                #v_mask=v[mask_f,:]
                #V_mask=v[mask_f,:][:,mask_f]
                #mu,cov=self.post_het2(v,Gamma,v_mask,V_mask,y_mask)
                mu,cov=self.post_het2(v,Gamma,self.rep,self.output)
            #f_mu=np.random.default_rng().multivariate_normal(mean=mu,cov=cov,check_valid='ignore')
            f_mu=fmvn_mu(mu,cov)
            return f_mu
        
    def posterior_vecch(self, idx, U_sp, ord, rev_ord):
        """Sampling from the conditional posterior distribution of the mean in heteroskedastic Gaussian likelihood under the Vecchia Approximation.
        """
        if idx==0:
            #if self.rep is None:
            invGamma = diags_array(np.exp(-self.input[:,1])[ord], format = 'csc')
            f_mu = self.post_het1_vecch(U_sp, invGamma, self.output[ord,0])[rev_ord]
            #else:
            #    invGamma = diags_array(np.exp(-self.input[:,1]), format = 'csc')
            #    Mt_invGamma_M = (self.rep_sp.transpose().dot(invGamma)).dot(self.rep_sp)
            #    Mt_invGamma_M = Mt_invGamma_M[ord,:][:,ord]
            #    invGammay = invGamma.dot(self.output[:,0])
            #    Mt_invGammay = self.rep_sp.transpose().dot(invGammay)[ord]
            #    f_mu = self.post_het2_vecch(U_sp, L_sp, Mt_invGamma_M, Mt_invGammay)[rev_ord]
            return f_mu
        
    @staticmethod
    def post_het1_vecch(U_sp, invGamma, y_mask):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are no repetitions
           in the training data under the Vecchia approximation.
        """
        invGammay = invGamma.dot(y_mask)
        sd = np.random.rand(len(y_mask))
        L_sp = U_sp.transpose().tocsr()
        # to be changed to spsolve_triangular when scipy updates with newer robust version
        #samp = spsolve(L_sp, sd)
        #intermediate = spsolve(U_sp, invGammay)
        #mu = spsolve(L_sp, intermediate)
        samp = forward_substitute(L_sp.data, L_sp.indices, L_sp.indptr, sd)
        intermediate = backward_substitute(U_sp.data, U_sp.indices, U_sp.indptr, invGammay)
        mu = forward_substitute(L_sp.data, L_sp.indices, L_sp.indptr, intermediate)
        f = mu + samp
        return f
    
    #@staticmethod
    #def post_het2_vecch(U_sp_post, L_sp, invGamma, invGammay):
    #    """Calculate the conditional posterior mean and covariance of the mean 
    #       of the heteroskedastic Gaussian likelihood when there are repetitions
    #       in the training data under the Vecchia approximation.
    #    """
    #    sd = np.random.rand(len(invGammay))
    #    L_sp_post = U_sp_post.transpose()
    #    U_sp = L_sp.transpose()
    #    samp = spsolve(L_sp_post, sd)
    #    intermediate_P_invGammay = spsolve(U_sp_post, invGammay)
    #    P_invGammay = spsolve(L_sp_post, intermediate_P_invGammay)
    #    Gamma_P_Gammay = invGamma.dot(P_invGammay)
    #    Gammay_Gamma_P_Gammay = invGammay - Gamma_P_Gammay
    #    intermediate_mu = spsolve(U_sp, Gammay_Gamma_P_Gammay)
    #    mu = spsolve(L_sp, intermediate_mu)
    #    f = mu + samp
    #    return f
        
    #@staticmethod
    #def post_het1_vecch(L_sp, invGamma, y_mask):
    #    """Calculate the conditional posterior mean and covariance of the mean 
    #       of the heteroskedastic Gaussian likelihood when there are no repetitions
    #       in the training data under the Vecchia approximation.
    #    """
    #    P = (L_sp.transpose()).dot(L_sp)
    #    invGammaP = invGamma + P
    #    invGammay = invGamma.dot(y_mask)
    #    cg_sampler = HetroSampler(invGammaP, L_sp, invGammay)
    #    f = cg_sampler.sample()
    #    return f
    
    #@staticmethod
    #def post_het2_vecch(L_sp, invGamma, invGammay):
    #    """Calculate the conditional posterior mean and covariance of the mean 
    #       of the heteroskedastic Gaussian likelihood when there are repetitions
    #       in the training data under the Vecchia approximation.
    #    """
    #    P = (L_sp.transpose()).dot(L_sp)
    #    invGammaP = invGamma + P
    #    cg_sampler = HetroSamplerRep(invGamma, invGammaP, L_sp, invGammay)
    #    f = cg_sampler.sample()
    #    return f

    @staticmethod
    def post_het1(v,Gamma,y_mask):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are no repetitions
           in the training data.
        """
        L=cholesky(Gamma+v,lower=True,check_finite=False)
        #mu=np.sum(v*cho_solve((L, True), y_mask, check_finite=False),axis=1)
        mu=(v@cho_solve((L, True), y_mask, check_finite=False)).flatten()
        cov=v@cho_solve((L, True), Gamma, check_finite=False)
        return mu, cov

    @staticmethod
    def post_het2(v,Gamma,mask_f,y_mask):
        """Calculate the conditional posterior mean and covariance of the mean 
           of the heteroskedastic Gaussian likelihood when there are repetitions
           in the training data.
        """
        L=cholesky(v,lower=True,check_finite=False)
        L_mask=L[mask_f,:]
        v_mask=v[mask_f,:]
        LGammaInv=L_mask.T*(1/Gamma)
        LGammaInvL_I=LGammaInv@L_mask+np.eye(len(L))
        LL = cholesky(LGammaInvL_I,lower=True,check_finite=False)
        LGammaInvY=LGammaInv@y_mask
        LGammaInvv=LGammaInv@v_mask
        vGamma=v_mask.T*(1/Gamma)
        vGammaInvY=vGamma@y_mask
        vGammav=vGamma@v_mask
        #mu=np.sum(v_mask.T*cho_solve((L, True), y_mask, check_finite=False),axis=1)
        mu=(vGammaInvY-LGammaInvv.T@cho_solve((LL, True), LGammaInvY, check_finite=False)).flatten()
        cov=v-vGammav+LGammaInvv.T@cho_solve((LL, True), LGammaInvv, check_finite=False)
        return mu, cov    

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
        y,mu,sigma=(self.output).flatten(),np.exp(self.input[:,0]),np.exp(self.input[:,1])
        llik=loggamma(y+1/sigma)-loggamma(1/sigma)-loggamma(y+1)+y*np.log(sigma*mu)-(y+1/sigma)*np.log(1+sigma*mu)
        llik=np.sum(llik)
        return llik

    @staticmethod
    def pllik(y,f):
        mu,sigma=np.exp(f[:,:,[0]]),np.exp(f[:,:,[1]])
        pllik=loggamma(y+1/sigma)-loggamma(1/sigma)-loggamma(y+1)+y*np.log(sigma*mu)-(y+1/sigma)*np.log(1+sigma*mu)
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


        