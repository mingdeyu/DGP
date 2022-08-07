import numpy as np
from numpy.random import randn, uniform, standard_t
from math import sqrt, pi
from scipy.optimize import minimize, Bounds
from scipy.linalg import cho_solve, pinvh
from scipy.spatial.distance import pdist, squareform
from .functions import gp, link_gp, pdist_matern_one, pdist_matern_multi, pdist_matern_coef, fod_exp, Z_fct

class kernel:
    """
    Class that defines the GPs in the DGP hierarchy.

    Args:
        length (ndarray): a numpy 1d-array, whose length equals to:

                1. either one if the lengthscales in the kernel function are assumed same across input dimensions; or
                2. the total number of input dimensions, which is the sum of the number of feeding GPs 
                   in the last layer (defined by the argument **input_dim**) and the number of connected global
                   input dimensions (defined by the argument **connect**), if the lengthscales in the kernel function 
                   are assumed different across input dimensions.
        scale (float, optional): the variance of a GP. Defaults to `1`.
        nugget (float, optional): the nugget term of a GP. Defaults to `1e-6`.
        name (str, optional): kernel function to be used. Either `sexp` for squared exponential kernel or
            `matern2.5` for Matern2.5 kernel. Defaults to `sexp`.
        prior_name (str, optional): prior options for the lengthscales and nugget term. Either gamma (`ga`) or inverse gamma (`inv_ga`) distribution for 
            the lengthscales and nugget term. Set `None` to disable the prior. Defaults to `ga`.
        prior_coef (ndarray, optional): a numpy 1d-array that contains two values specifying the shape and rate 
            parameters of gamma prior, shape and scale parameters of inverse gamma prior. Defaults to ``np.array([1.6,0.3])``.
        nugget_est (bool, optional): set to `True` to estimate nugget term or to `False` to fix the nugget term as specified
            by the argument **nugget**. If set to `True`, the value set to the argument **nugget** is used as the initial
            value. Defaults to `False`.
        scale_est (bool, optional): set to `True` to estimate the variance or to `False` to fix the variance as specified
            by the argument **scale**. Defaults to `False`.
        input_dim (ndarray, optional): a numpy 1d-array that contains either

                1. the indices of GPs in the feeding layer whose outputs feed into the GP; or
                2. the indices of dimensions in the global input if the GP is in the first layer. 
            When set to `None`, 
            
                1. all outputs from GPs in the feeding layer; or 
                2. all global input dimensions feed into the GP. 
            Defaults to `None`.
        connect (ndarray, optional): a numpy 1d-array that contains the indices of dimensions in the global
            input connecting to the GP as additional input dimensions to the input obtained from the output of
            GPs in the feeding layer (as determined by the argument **input_dim**). When set to `None`, no global input
            connection is implemented. Defaults to `None`. When the kernel class is used in GP/DGP emulators for linked
            emulation and some input dimensions to the computer models are not connected to some feeding computer models, 
            set **connect** to a 1d-array of indices of these external global input dimensions, and accordingly, set 
            **input_dim** to a 1d-array of indices of the remaining input dimensions that are connected to the feeding 
            computer models.                   

    Attributes:
        type (str): identifies that the kernel is a GP.
        g (function): a function giving the log probability density function of gamma or inverse gamma distribution 
            ignoring the constant part.
        gfod (function): a function giving the first order derivative of **g** with respect to the log-transformed 
            lengthscales and nugget. 
        para_path (ndarray): a numpy 2d-array that contains the trace of model parameters. Each row is a 
            parameter estimate produced by one SEM iteration. The model parameters in each row are ordered as 
            follow: ``np.array([scale estimate, lengthscale estimate (whose length>=1), nugget estimate])``.
        global_input (ndarray): a numpy 2d-array that contains the connect global input dimensions determined 
            by the argument **connect**. The value of the attribute is assigned during the initialisation of 
            :class:`.dgp` class. If **connect** is set to `None`, this attribute is also `None`. 
        input (ndarray): a numpy 2d-array (each row as a data point and each column as a data dimension) that 
            contains the input training data (according to the argument **input_dim**) to the GP. The value of 
            this attribute is assigned during the initialisation of :class:`.dgp` class. 
        output (ndarray): a numpy 2d-array with only one column that contains the output training data to the GP.
            The value of this attribute is assigned during the initialisation of :class:`.dgp` class.
        rep (ndarray): a numpy 1d-array used to re-construct repetitions in the data according to the repetitions 
            in the global input, i.e., rep is assigned during the initialisation of :class:`.dgp` class if one input position 
            has multiple outputs. Otherwise, it is `None`. Defaults to `None`. 
        Rinv (ndarray): a numpy 2d-array that stores the inversion of correlation matrix. Defaults to `None`.
        Rinv_y (ndarray): a numpy 2d-array that stores the product of correlation matrix inverse and the output Y. Defaults to `None`.
        rff (bool): indicates weather random Fourier features are used. Defaults to `None`.
        D (int): the dimension of input data to the GP node. Defaults to `None`.
        M (int): the number of features in random Fourier approximation. Defaults to `None`.
        W (ndarray): a 2d-array (M, D) sampled to construct the Fourier approximation to the kernel matrix. Defaults to `None`.
        b (ndarray): a 1d-array (D,) sampled to construct the Fourier approximation to the kernel matrix. Defaults to `None`.
    """

    def __init__(self, length, scale=1., nugget=1e-6, name='sexp', prior_name='ga', prior_coef=np.array([1.6,0.3]), nugget_est=False, scale_est=False, input_dim=None, connect=None):
        self.type='gp'
        self.length=length
        self.scale=np.atleast_1d(scale)
        self.nugget=np.atleast_1d(nugget)
        self.name=name
        self.prior_name=prior_name
        self.prior_coef=prior_coef
        if self.prior_name=='ga':
            self.g=lambda x: (self.prior_coef[0]-1)*np.log(x)-self.prior_coef[1]*x
            self.gfod=lambda x: (self.prior_coef[0]-1)-self.prior_coef[1]*x
        elif self.prior_name=='inv_ga':
            self.g=lambda x: -(self.prior_coef[0]+1)*np.log(x)-self.prior_coef[1]/x
            self.gfod=lambda x: -(self.prior_coef[0]+1)+self.prior_coef[1]/x
        self.nugget_est=nugget_est
        self.scale_est=scale_est
        self.input_dim=input_dim
        self.connect=connect
        self.para_path=np.atleast_2d(np.concatenate((self.scale,self.length,self.nugget)))
        self.global_input=None
        self.input=None
        self.output=None
        self.rep=None
        self.Rinv=None
        self.Rinv_y=None
        self.rff=None
        self.D=None
        self.M=None
        self.W=None
        self.b=None

    def sample_basis(self):
        """Sample **W** and **b** to construct random Fourier approximations to correlation matrices.
        """
        if self.name=='sexp':
            self.W=sqrt(2)*randn(self.M,self.D)
        elif self.name=='matern2.5':
            self.W=standard_t(5,size=(self.M,self.D))
        self.b=uniform(0,2*pi,size=self.M)

    def log_t(self):
        """Log transform the model parameters (lengthscales and nugget).

        Returns:
            ndarray: a numpy 1d-array of log-transformed model parameters
        """
        if self.nugget_est:
            log_theta=np.log(np.concatenate((self.length,self.nugget)))
        else:
            log_theta=np.log(self.length)
        return log_theta

    def update(self,log_theta):
        """Update the model parameters (scale, lengthscales and nugget).

        Args:
            log_theta (ndarray): optimised numpy 1d-array of log-transformed lengthscales and nugget.
        """
        theta=np.exp(log_theta)
        if self.nugget_est:
            self.length=theta[0:-1]
            self.nugget=theta[[-1]]
        else:
            self.length=theta
        if self.scale_est:
            K=self.k_matrix()
            L=np.linalg.cholesky(K)
            YKinvY=(self.output).T@cho_solve((L, True), self.output, check_finite=False)
            new_scale=YKinvY/len(self.output)
            self.scale=new_scale.flatten()
            
    def k_matrix(self,fod_eval=False):
        """Compute the correlation matrix and/or first order derivatives of the correlation matrix wrt log-transformed lengthscales and nugget.
        
        Args:
            fod_eval (bool): indicates if the gradient information is also computed along with the correlation
                matrix. Defaults to `False`. 

        Returns:
            ndarray_or_tuple: 
                1. If **fod_eval** = `False`, a numpy 2d-array *K* is returned as the correlation matrix.
                2. If **fod_eval** = `True`, a tuple is returned. It includes *K* and fod, a numpy 3d-array that contains the first order derivatives of the correlation matrix 
                   wrt log-transformed lengthscales and nugget. The length of the array equals to the total number 
                   of model parameters (i.e., the total number of lengthscales and nugget).
        """
        n=len(self.input)
        if self.global_input is not None:
            X=np.concatenate((self.input, self.global_input),1)
        else:
            X=self.input
        X_l=X/self.length
        if self.name=='sexp':
            dists = pdist(X_l, metric="sqeuclidean")
            K = squareform(np.exp(-dists))
            if fod_eval:
                if len(self.length)==1:
                    fod=np.expand_dims(squareform(2*dists)*K,axis=0)
                else:
                    fod=fod_exp(X_l,K)
        elif self.name=='matern2.5':
            if fod_eval:
                K=squareform(np.exp(-np.sqrt(5)*pdist(X_l, metric="minkowski",p=1)))
                if len(self.length)==1:
                    coef1, coef2 = pdist_matern_one(X_l)
                else:
                    coef1, coef2 = pdist_matern_multi(X_l)
                K*=coef1
                fod=coef2*K
            else:
                K=np.exp(-np.sqrt(5)*pdist(X_l, metric="minkowski",p=1))
                K*=pdist_matern_coef(X_l)
                K=squareform(K)
        if fod_eval and self.nugget_est:
            nugget_fod=np.expand_dims(self.nugget*np.eye(n),0)
            fod=np.concatenate((fod,nugget_fod),axis=0)
        np.fill_diagonal(K, 1+self.nugget)
        if fod_eval:
            return K, fod
        else:
            return K
    
    def log_prior(self):
        """Compute the value of log priors specified to the lengthscales and nugget. 

        Returns:
            ndarray: a numpy 1d-array giving the sum of log priors of the lengthscales and nugget. 
        """
        lp=np.sum(self.g(self.length),keepdims=True)
        if self.nugget_est:
            lp+=self.g(self.nugget)
        return lp

    def log_prior_fod(self):
        """Compute the first order derivatives of log priors wrt the log-transformed lengthscales and nugget.

        Returns:
            ndarray: a numpy 1d-array (whose length equal to the total number of lengthscales and nugget)
            giving the first order derivatives of log priors wrt the log-transformed lengthscales and nugget.
        """
        fod=self.gfod(self.length)
        if self.nugget_est:
            fod=np.concatenate((fod,self.gfod(self.nugget)))
        return fod

    def llik(self,x):
        """Compute the negative log-likelihood function of the GP.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            ndarray: a numpy 1d-array giving negative log-likelihood.
        """
        self.update(x)
        n=len(self.output)
        K=self.k_matrix()
        L=np.linalg.cholesky(K)
        logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        YKinvY=(self.output).T@cho_solve((L, True), self.output, check_finite=False)
        if self.scale_est:
            scale=YKinvY/n
            neg_llik=0.5*(logdet+n*np.log(scale))
        else:
            neg_llik=0.5*(logdet+YKinvY/self.scale) 
        neg_llik=neg_llik.flatten()
        if self.prior_name!=None:
            neg_llik=neg_llik-self.log_prior()
        return neg_llik
    
    def llik_rff(self,x):
        """Compute the negative log-likelihood function of the GP under RFF.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            ndarray: a numpy 1d-array giving negative log-likelihood.
        """
        self.update(x)
        n=len(self.output)
        if self.connect is not None:
            X=np.concatenate((self.input,self.global_input),1)
        else:
            X=self.input
        Z=Z_fct(X,self.W,self.b,self.length,self.M)
        cov=np.dot(Z.T,Z)+self.nugget*np.identity(self.M)
        L=np.linalg.cholesky(cov)
        logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        Zt_y=np.dot(Z.T,self.output)
        quad=np.dot(self.output.T,self.output)-np.sum(Zt_y*cho_solve((L, True), Zt_y, check_finite=False))
        if self.scale_est:
            scale=quad/(n*self.nugget)
            neg_llik=0.5*(logdet+n*np.log(scale)+(n-self.M)*np.log(self.nugget))
        else:
            neg_llik=0.5*(logdet+quad/(self.scale*self.nugget)+(n-self.M)*np.log(self.nugget)) 
        neg_llik=neg_llik.flatten()
        if self.prior_name!=None:
            neg_llik=neg_llik-self.log_prior()
        return neg_llik

    def llik_der(self,x):
        """Compute first order derivatives of the negative log-likelihood function wrt log-transformed model parameters.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget.

        Returns:
            ndarray: a numpy 1d-array (whose length equal to the total number of lengthscales and nugget)
            that contains first order derivatives of the negative log-likelihood function wrt log-transformed 
            lengthscales and nugget.
        """
        self.update(x)
        n=len(self.output)
        K,Kt=self.k_matrix(fod_eval=True)
        KinvKt=np.linalg.solve(K,Kt)
        tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
        L=np.linalg.cholesky(K)
        KinvY=cho_solve((L, True), self.output, check_finite=False)
        YKinvKtKinvY=((self.output).T@KinvKt@KinvY).flatten()
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if self.scale_est:
            YKinvY=(self.output).T@KinvY
            scale=(YKinvY/n).flatten()
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2/self.scale
        if self.prior_name!=None:
            neg_St=neg_St-self.log_prior_fod()
        return neg_St

    def log_likelihood_func(self):
        cov=self.scale*self.k_matrix()
        L=np.linalg.cholesky(cov)
        logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        quad=(self.output).T@cho_solve((L, True), self.output, check_finite=False)
        llik=-0.5*(logdet+quad)
        return llik

    def log_likelihood_func_rff(self):
        """Compute Gaussian log-likelihood function using random Fourier features (RFF).
        """
        if self.connect is not None:
            X=np.concatenate((self.input,self.global_input),1)
        else:
            X=self.input
        Z=Z_fct(X,self.W,self.b,self.length,self.M)
        cov=np.dot(Z.T,Z)+self.nugget*np.identity(self.M)
        L=np.linalg.cholesky(cov)
        logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        Zt_y=np.dot(Z.T,self.output)
        quad=-np.sum(Zt_y*cho_solve((L, True), Zt_y, check_finite=False))/(self.scale*self.nugget)
        #quad=-(Zt_y.T@cho_solve((L, True), Zt_y, check_finite=False))/(self.scale*self.nugget)
        llik=-0.5*(logdet+quad)
        return llik

    def maximise(self, method='L-BFGS-B'):
        """Optimise and update model parameters by minimising the negative log-likelihood function.

        Args:
            method (str, optional): optimisation algorithm. Defaults to `L-BFGS-B`.
        """
        initial_theta_trans=self.log_t()
        if self.nugget_est:
            if self.rff:
                lb=np.concatenate((-5.*np.ones(len(initial_theta_trans)-1),np.log([1e-8])))
                ub=5.*np.ones(len(initial_theta_trans))
                bd=Bounds(lb, ub)
                _ = minimize(self.llik_rff, initial_theta_trans, method=method, bounds=bd, options={'maxiter': 100, 'maxfun': 125})
            else:
                lb=np.concatenate((-np.inf*np.ones(len(initial_theta_trans)-1),np.log([1e-8])))
                ub=np.inf*np.ones(len(initial_theta_trans))
                bd=Bounds(lb, ub)
                _ = minimize(self.llik, initial_theta_trans, method=method, jac=self.llik_der, bounds=bd, options={'maxiter': 100, 'maxfun': 125})
        else:
            if self.rff:
                lb=-5.*np.ones(len(initial_theta_trans))
                ub=5.*np.ones(len(initial_theta_trans))
                bd=Bounds(lb, ub)
                _ = minimize(self.llik_rff, initial_theta_trans, method=method, bounds=bd, options={'maxiter': 100, 'maxfun': 125})
            else:
                _ = minimize(self.llik, initial_theta_trans, method=method, jac=self.llik_der, options={'maxiter': 100, 'maxfun': 125})
        self.add_to_path()
        
    def add_to_path(self):
        """Add updated model parameter estimates to the class attribute **para_path**.
        """
        para=np.concatenate((self.scale,self.length,self.nugget))
        self.para_path=np.vstack((self.para_path,para))

    def gp_prediction(self,x,z):
        """Make GP predictions. 

        Args:
            x (ndarray): a numpy 2d-array that contains the input testing data (whose rows correspond to testing
                data points and columns correspond to testing data dimensions) with the number of columns same as 
                the **input** attribute.
            z (ndarray): a numpy 2d-array that contains additional input testing data (with the same number of 
                columns of the **global_input** attribute) from the global testing input if the argument **connect** 
                is not `None`. Set to None if the argument **connect** is `None`. 

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions. 
        """
        m,v=gp(x,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.scale,self.length,self.nugget,self.name)
        return m,v

    def linkgp_prediction(self,m,v,z,nb_parallel):
        """Make linked GP predictions. 

        Args:
            m (ndarray): a numpy 2d-array that contains predictive means of testing outputs from the GPs in the last 
                layer. The number of rows equals to the number of testing positions and the number of columns equals to the 
                length of the argument **input_dim**. If the argument **input_dim** is `None`, then the number of columns equals 
                to the number of GPs in the last layer.
            v (ndarray): a numpy 2d-array that contains predictive variances of testing outputs from the GPs in the last 
                layer. It has the same shape of **m**.
            z (ndarray): a numpy 2d-array that contains additional input testing data (with the same number of 
                columns of the **global_input** attribute) from the global testing input if the argument **connect** 
                is not `None`. Set to `None` if the argument **connect** is `None`. 
            nb_parallel (bool): whether to use *Numba*'s multi-threading to accelerate the predictions.

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions (that are 
            represented by predictive means and variances).
        """
        m,v=link_gp(m,v,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.scale,self.length,self.nugget,self.name,nb_parallel)
        return m,v

    def linkgp_prediction_full(self,m,v,m_z,v_z,z,nb_parallel):
        """Make linked GP predictions with additional input also generated by GPs/DGPs. 

        Args:
            m (ndarray): a numpy 2d-array that contains predictive means of testing outputs from the GPs in the last 
                layer. The number of rows equals to the number of testing positions and the number of columns equals to the 
                length of the argument **input_dim**. If the argument **input_dim** is `None`, then the number of columns equals 
                to the number of GPs in the last layer.
            v (ndarray): a numpy 2d-array that contains predictive variances of testing outputs from the GPs in the last 
                layer. It has the same shape of **m**.
            m_z (ndarray): a numpy 2d-array that contains predictive means of additional input testing data from GPs.
            v_z (ndarray): a numpy 2d-array that contains predictive variances of additional input testing data from GPs.
            z (ndarray): a numpy 2d-array that contains additional input testing data from the global testing input that are
                not from GPs. Set to `None` if the argument **connect** is None. 
            nb_parallel (bool): whether to use *Numba*'s multi-threading to accelerate the predictions.

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions (that are 
            represented by predictive means and variances).
        """
        m=np.concatenate((m,m_z),axis=1)
        v=np.concatenate((v,v_z),axis=1)
        idx1=np.arange(np.shape(m_z)[1])
        idx2=np.arange(np.shape(m_z)[1],np.shape(self.global_input)[1])
        overall_input=np.concatenate((self.input,self.global_input[:,idx1]),axis=1)
        m,v=link_gp(m,v,z,overall_input,self.global_input[:,idx2],self.Rinv,self.Rinv_y,self.scale,self.length,self.nugget,self.name,nb_parallel)
        return m,v

    def compute_stats(self):
        """Compute and store key statistics for the GP predictions
        """
        R=self.k_matrix()
        #U, s, Vh = np.linalg.svd(R)
        #self.Rinv=Vh.T@np.diag(s**-1)@U.T
        #L=np.linalg.cholesky(R)
        #self.Rinv_y=cho_solve((L, True), self.output, check_finite=False)
        self.Rinv=pinvh(R,check_finite=False)
        self.Rinv_y=np.dot(self.Rinv,self.output)

def combine(*layers):
    """Combine layers into one list as a DGP structure.

    Args:
        layers (list): a sequence of lists, each of which contains the GPs (defined by the :class:`.kernel` class) in that layer.

    Returns:
        list: a list of layers defining the DGP structure.
    """
    all_layer=[]
    for layer in layers:
        all_layer.append(layer)
    return all_layer