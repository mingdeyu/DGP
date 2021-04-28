import numpy as np
from math import sqrt
from scipy.optimize import minimize, Bounds
from functions import gp, link_gp

class kernel:
    """
    Class that defines the GPs in the DGP hierarchy.

        Args:
            length (ndarray): a numpy 1d-array, whose length equals to:
                1. one if the lengthscales in the kernel function are assumed same across input dimensions;
                2. the total number of input dimensions, which is the sum of the number of feeding GPs 
                in the last layer (defined by the argument 'input_dim') and the number of connected global
                input dimensions (defined by the argument 'connect'), if the lengthscales in the kernel function 
                are assumed different across input dimensions.
            scale (float, optional): the variance of a GP. Defaults to 1..
            nugget (float, optional): the nugget term of a GP. Defaults to 1e-8.
            name (str, optional): kernel function to be used. Either 'sexp' for squared exponential kernel or
                'matern2.5' for Matern2.5 kernel. Defaults to 'sexp'.
            prior (ndarray, optional): a numpy 1d-array that contains two values specifying the priors specified
                for the lengthscales and nugget. Defaults to np.array([0.3338,0.0835]).
            nugget_est (int, optional): set to 1 to estimate nugget term or to 0 to fix the nugget term as specified
                by the argument 'nugget'. If set to 1, the value set to the argument 'nugget' is used as the initial
                value. Defaults to 0.
            scale_est (int, optional): set to 1 to estimate the variance or to 0 to fix the variance as specified
                by the argument 'scale'. Defaults to 0.
            input_dim (ndarray, optional): a numpy 1d-array that contains the indices of GPs in the last layer
                   whose outputs (or the indices of dimensions in the global input if the GP is in the first layer)
                   feed into the GP. When set to None, all outputs from GPs of last layer (or all global input 
                   dimensions) feed into the GP. Defaults to None.
            connect (ndarray, optional): a numpy 1d-array that contains the indices of dimensions in the global
                input connecting to the GP as additional input dimensions to the input obtained from the output of
                GPs in the last layer (as determined by the argument 'input_dim'). When set to None, no global input
                connection is implemented. Defaults to None.

        Attributes:
            para_path (ndarray): a numpy 2d-array that contains the trace of model parameters. Each row is a 
                parameter estimate produced by one SEM iteration. The model parameters in each row are ordered as 
                follow: np.array([scale estimate, lengthscale estimate (whose length>=1), nugget estimate]).
            global_input (ndarray): a numpy 2d-array that contains the connect global input dimensions determined 
                by the argument 'connect'. The value of the attribute is assigned during the initialisation of 
                'dgp' class. If 'connect' is set to None, this attribute is also None.
            input (ndarray): a numpy 2d-array (each row as a data point and each column as a data dimension) that 
                contains the input training data (according to the argument 'input_dim') to the GP. The value of 
                this attribute is assigned during the initialisation of 'dgp' class.
            output (ndarray): a numpy 2d-array with only one column that contains the output training data to the GP.
                The value of this attribute is assigned during the initialisation of 'dgp' class.
            missingness (ndarray): a numpy 1d-array of bool that indicates the missingness in the output attributes.
                If a cell is True, then the corresponding cell in the output attribute needs to be imputed. The value 
                of this attribute is assigned during the initialisation of 'dgp' class. 
        """

    def __init__(self, length, scale=1., nugget=1e-8, name='sexp', prior=np.array([0.3338,0.0835]), nugget_est=0, scale_est=0, input_dim=None, connect=None):
        self.length=length
        self.scale=np.atleast_1d(scale)
        self.nugget=np.atleast_1d(nugget)
        self.name=name
        self.prior=prior
        self.nugget_est=nugget_est
        self.scale_est=scale_est
        self.input_dim=input_dim
        self.connect=connect
        self.para_path=np.concatenate((self.scale,self.length,self.nugget))
        self.global_input=None
        self.input=None
        self.output=None
        self.missingness=None

    def log_t(self):
        """Log transform the model paramters (lengthscales and nugget).

        Returns:
            ndarray: a numpy 1d-array of log-transformed model paramters
        """
        if self.nugget_est==1:
            log_theta=np.log(np.concatenate((self.length,self.nugget)))
        else:
            log_theta=np.log(self.length)
        return log_theta

    def update(self,log_theta):
        """Update the model paramters (scale, lengthscales and nugget).

        Args:
            log_theta (ndarray): optimised numpy 1d-array of log-transformed lengthscales and nugget.
        """
        theta=np.exp(log_theta)
        if self.nugget_est==1:
            self.length=theta[0:-1]
            self.nugget=theta[[-1]]
        else:
            self.length=theta
        if self.scale_est==1:
            K=self.k_matrix()
            KinvY=np.linalg.solve(K,self.output)
            YKinvY=(self.output).T@KinvY
            new_scale=YKinvY/len(self.output)
            self.scale=new_scale.flatten()
            
    def k_matrix(self):
        """Compute the correlation matrix.

        Returns:
            ndarray: a numpy 2d-array as the correlation matrix.
        """
        n=len(self.input)
        if np.any(self.connect!=None):
            X=np.concatenate((self.input, self.global_input),1)
        else:
            X=self.input
        X_l=X/self.length
        if self.name=='sexp':
            L=np.sum(X_l**2,1).reshape([-1,1])
            dis2=np.abs(L-2*X_l@X_l.T+L.T)
            K=np.exp(-dis2)
        elif self.name=='matern2.5':
            X_l=np.expand_dims(X_l.T,axis=2)
            dis=np.abs(X_l-X_l.transpose([0,2,1]))
            K_1=np.prod(1+sqrt(5)*dis+5/3*dis**2,0)
            K_2=np.exp(-sqrt(5)*np.sum(dis,0))
            K=K_1*K_2
        return K+self.nugget*np.eye(n)

    def k_fod(self):
        """Compute first order derivatives of the correlation matrix wrt log-transformed lengthscales and nugget.

        Returns:
            ndarray: a numpy 3d-array that contains the first order derivatives of the correlation matrix 
                wrt log-transformed lengthscales and nugget. The length of the array equals to the total number 
                of model parameters (i.e., the total number of lengthscales and nugget). 
        """
        n=len(self.input)
        if np.any(self.connect!=None):
            X=np.concatenate((self.input, self.global_input),1)
        else:
            X=self.input
        X_l=X/self.length
        
        X_li=np.expand_dims(X_l.T,axis=2)
        disi=np.abs(X_li-X_li.transpose([0,2,1]))
        if self.name=='sexp':
            dis2=np.sum(disi**2,axis=0,keepdims=True)
            K=np.exp(-dis2)
            if len(self.length)==1:
                fod=2*dis2*K
            else:
                fod=2*(disi**2)*K
        elif self.name=='matern2.5':
            K_1=np.prod(1+sqrt(5)*disi+5/3*disi**2,axis=0,keepdims=True)
            K_2=np.exp(-sqrt(5)*np.sum(disi,axis=0,keepdims=True))
            K=K_1*K_2
            coefi=(disi**2)*(1+sqrt(5)*disi)/(1+sqrt(5)*disi+5/3*disi**2)
            if len(self.length)==1:
                fod=5/3*np.sum(coefi,axis=0,keepdims=True)*K
            else:
                fod=5/3*coefi*K
        if self.nugget_est==1:
            nugget_fod=np.expand_dims(self.nugget*np.eye(n),0)
            fod=np.concatenate((fod,nugget_fod),axis=0)
        return fod
    
    def log_prior(self):
        """Compute the value of log priors specified to the lengthscales and nugget. 

        Returns:
            ndarray: a numpy 1d-array giving the sum of log priors of the lengthscales and nugget. 
        """
        lp=np.sum(2*self.prior[0]*np.log(self.length)-self.prior[1]*self.length**2,keepdims=True)
        if self.nugget_est==1:
            lp+=2*self.prior[0]*np.log(self.nugget)-self.prior[1]*self.nugget**2
        return lp

    def log_prior_fod(self):
        """Compute the first order derivatives of log priors wrt the log-tranformed lengthscales and nugget.

        Returns:
            ndarray: a numpy 1d-array (whose length equal to the total number of lengthscales and nugget)
                giving the first order derivatives of log priors wrt the log-tranformed lengthscales and nugget.
        """
        fod=2*(self.prior[0]-self.prior[1]*self.length**2)
        if self.nugget_est==1:
            fod=np.concatenate((fod,2*(self.prior[0]-self.prior[1]*self.nugget**2)))
        return fod

    def llik(self,x):
        """Compute the negative log-likelihood function of the GP.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model paramters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            ndarray: a numpy 1d-array giving negative log-likelihood.
        """
        self.update(x)
        n=len(self.output)
        K=self.k_matrix()
        _,logdet=np.linalg.slogdet(K)
        KinvY=np.linalg.solve(K,self.output)
        YKinvY=(self.output).T@KinvY
        if self.scale_est==1:
            scale=YKinvY/n
            neg_llik=0.5*(logdet+n*np.log(scale))
        else:
            neg_llik=0.5*(logdet+YKinvY) 
        neg_llik=neg_llik.flatten()
        if np.any(self.prior!=None):
            neg_llik=neg_llik-self.log_prior()
        return neg_llik

    def llik_der(self,x):
        """Compute first order derivatives of the negative log-likelihood function wrt log-tranformed model parameters.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model paramters: 
                log-transformed lengthscales followed by the log-transformed nugget.

        Returns:
            ndarray: a numpy 1d-array (whose length equal to the total number of lengthscales and nugget)
                that contains first order derivatives of the negative log-likelihood function wrt log-tranformed 
                lengthscales and nugget.
        """
        self.update(x)
        n=len(self.output)
        K=self.k_matrix()
        Kt=self.k_fod()
        KinvKt=np.linalg.solve(K,Kt)
        tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
        KinvY=np.linalg.solve(K,self.output)
        YKinvKtKinvY=((self.output).T@KinvKt@KinvY).flatten()
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if self.scale_est==1:
            YKinvY=(self.output).T@KinvY
            scale=(YKinvY/n).flatten()
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2
        if np.any(self.prior!=None):
            neg_St=neg_St-self.log_prior_fod()
        return neg_St

    def maximise(self, method='L-BFGS-B'):
        """Optimise and update model parameters by minimising the negative log-likelihood function.

        Args:
            method (str, optional): optimisation algorithm. Defaults to 'L-BFGS-B'.
        """
        initial_theta_trans=self.log_t()
        if self.nugget_est==1:
            lb=np.concatenate((-np.inf*np.ones(len(initial_theta_trans)-1),np.log([1e-8])))
            ub=np.inf*np.ones(len(initial_theta_trans))
            bd=Bounds(lb, ub)
            _ = minimize(self.llik, initial_theta_trans, method=method, jac=self.llik_der, bounds=bd)
        else:
            _ = minimize(self.llik, initial_theta_trans, method=method, jac=self.llik_der)
        self.add_to_path()
        
    def add_to_path(self):
        """Add updated model parameter estimates to the class attribute 'para_path'.
        """
        para=np.concatenate((self.scale,self.length,self.nugget))
        self.para_path=np.vstack((self.para_path,para))

    def gp_prediction(self,x,z):
        """Make GP predictions. 

        Args:
            x (ndarray): a numpy 2d-array that contains the input testing data (whose rows correspond to testing
                data points and columns correspond to testing data dimensions) with the number of columns same as 
                the 'input' attribute.
            z (ndarray): a numpy 2d-array that contains additional input testing data (with the same number of 
                columns of the 'global_input' attribute) from the global testing input if the argument 'connect' 
                is not None. Set to None if the argument 'connect' is None. 

        Returns:
            list: a list of two 1d-arrays giving the means and variances at the testing input data positions. 
        """
        m,v=gp(x,z,self.input,self.global_input,self.output,self.scale,self.length,self.nugget,self.name)
        return m,v

    def linkgp_prediction(self,m,v,z):
        """Make linked GP predictions. 

        Args:
            m (ndarray): a numpy 2d-array that contains predictive means of testing outputs from the GPs in the last 
                layer. The number of rows equals to the number of testing positions and the number of columns equals to the 
                length of the argument 'input_dim'. If the argument 'input_dim' is None, then the number of columns equals 
                to the number of GPs in the last layer.
            v (ndarray): a numpy 2d-array that contains predictive variances of testing outputs from the GPs in the last 
                layer. It has the same shape of 'm'.
            z (ndarray): a numpy 2d-array that contains additional input testing data (with the same number of 
                columns of the 'global_input' attribute) from the global testing input if the argument 'connect' 
                is not None. Set to None if the argument 'connect' is None. 

        Returns:
            list: a list of two 1d-arrays giving the means and variances at the testing input data positions (that are 
                represented by predictive means and variances).
        """
        m,v=link_gp(m,v,z,self.input,self.global_input,self.output,self.scale,self.length,self.nugget,self.name)
        return m,v

def combine(*layers):
    """Combine layers into one list as a DGP structure.

    Args:
        *layers (list): a sequence of lists, each of which contains the GPs (defined by the kernel class) in that layer.

    Returns:
        list: a list of layers defining the DGP structure.
    """
    all_layer=[]
    for layer in layers:
        all_layer.append(layer)
    return all_layer