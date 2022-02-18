import numpy as np
from scipy.special import loggamma
from .functions import post_het1, post_het2

class Poisson:
    """Class to implement Poisson likelihood. It (and all likelihoods below) can only be added as the final
       layer of the DGP+likelihood model.

        Args:
            input_dim (ndarray, optional): a numpy 1d-array that contains the indices of GPs in the last 
            layer whose outputs feed into the likelihood node. When set to None, all outputs from GPs of 
            last layer feed into the likelihood node. Defaults to None.

        Attributes:
            type (str): identifies that the node is a likelihood node;
            input (ndarray): a numpy 2d-array (each row as a data point and each column as a likelihood parameter from the
                DGP part) that contains the input data (according to the argument 'input_dim') to the likelihood node. The value of 
                this attribute is assigned during the initialisation of 'dgp' class. 
            output (ndarray): a numpy 2d-array with only one column that contains the output data to the likelihood node.
                The value of this attribute is assigned during the initialisation of 'dgp' class.
            exact_post_idx (ndarray): a numpy 1d-array that indicates the indices of the likelihood parameters that allow closed-form
               conditional posterior distributions. Defaults to None.
            rep (ndarray): a numpy 1d-array used to re-construct repetitions in the data according to the repetitions in the global input,
                i.e., rep is assigned during the initialisation of 'dgp' class if one input position has multiple outputs. Otherwise, it is
                None. Defaults to None. 
    """
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='Poisson'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.last_layer_input=None
        self.missingness=None
        self.exact_post_idx=None
        self.rep=None

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
            y (ndarray): a numpy 3d-array of output data with shape (N,1,1), where N is the number of output data points.
            f (ndarray): a numpy 3d-array of sample points with shape (N,S,Q), where S is the number of sample points and 
                         Q is the number of parameters in the distribution (e.g., Q=1 for Poisson distribution).

        Returns:
            ndarray: a numpy 3d-array of log-likelihood for given f.
        """
        pllik=y*f-np.exp(f)-loggamma(y+1)
        return pllik
        
    def prediction(self,m,v):
        """Compute mean and variance of the DGP+Poisson model given the predictive
           mean and variance of DGP model for Poisson parameter.
        
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
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='Hetero'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.last_layer_input=None
        self.missingness=None
        self.exact_post_idx=np.array([0])
        self.rep=None

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
        
    def prediction(self,m,v):
        y_mean=m[:,0]
        y_var=np.exp(m[:,1]+v[:,1]/2)+v[:,0]
        return y_mean.flatten(),y_var.flatten()
    
    def sampling(self,f_sample):
        y_sample=np.random.normal(f_sample[:,0],np.sqrt(np.exp(f_sample[:,1])))
        return y_sample.flatten()

    def posterior(self,idx,mask,m,v):
        """Sampling from the conditional posterior distribution of the mean in heteroskedastic Gaussian likelihood.
        """
        if idx==0:
            if self.rep is None:
                Gamma=np.diag(np.exp(self.input[mask,1]))
                y_mask=self.output[mask,0]
                mu,cov=post_het1(m,v,Gamma,y_mask)
            else:
                Gamma=np.diag(np.exp(self.input[:,1]))
                y_mask=self.output[:,0]
                mask_f=self.rep
                v_mask=v[mask_f,:]
                V_mask=v[mask_f,:][:,mask_f]
                m_mask=m[mask_f]
                mu,cov=post_het2(m,v,Gamma,v_mask,V_mask,m_mask,y_mask)
            f_mu=np.random.default_rng().multivariate_normal(mean=mu,cov=cov,check_valid='ignore')
            return f_mu

class NegBin:
    def __init__(self, input_dim=None):
        self.type='likelihood'
        self.name='NegBin'
        self.input=None
        self.output=None
        self.input_dim=input_dim
        self.last_layer_input=None
        self.missingness=None
        self.exact_post_idx=None
        self.rep=None
    
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
    
    def prediction(self,m,v):
        y_mean=np.exp(m[:,0]+v[:,0]/2)
        y_var=np.exp(2*m[:,0]+v[:,0])*(np.exp(v[:,0])-1)+np.exp(m[:,0]+v[:,0]/2)+np.exp(m[:,1]+v[:,1]/2)*np.exp(2*m[:,0]+2*v[:,0])
        return y_mean.flatten(),y_var.flatten()
    
    def sampling(self,f_sample):
        p, k=1/(1+np.exp(f_sample[:,0]+f_sample[:,1])), np.exp(-f_sample[:,1])
        y_sample=np.random.negative_binomial(k,p)
        return y_sample.flatten()


        