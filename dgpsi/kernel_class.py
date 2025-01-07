import numpy as np
from numpy.linalg import LinAlgError, lstsq, matrix_rank
from scipy.optimize import minimize, Bounds
from scipy.linalg import cho_solve, pinvh, cholesky
from scipy.spatial.distance import pdist, squareform
from .functions import Pmatrix, gp, gp_non_parallel, link_gp, link_gp_non_parallel, pdist_matern_one, pdist_matern_multi, pdist_matern_coef, fod_exp, logdet_nb, trace_nb, g
from .vecchia import nn, vecchia_llik, vecchia_nllik, get_pred_nn, gp_vecch, gp_vecch_non_parallel, imp_pointers, imp_pointers_rep, link_gp_vecch, link_gp_vecch_non_parallel
from .utils import get_thread
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
        prior_name (str, optional): prior options for the lengthscales and nugget term. Either gamma (`ga`), inverse gamma (`inv_ga`) or the reference
            prior (`ref`) for the lengthscales and nugget term. Set `None` to disable the prior. Defaults to `ga`.
        prior_coef (ndarray, optional): if **prior_name** is either `ga` or `inv_ga`, it is a numpy 1d-array that contains two values specifying the shape 
            and rate parameters of gamma prior, or shape and scale parameters of inverse gamma prior. If  **prior_name** is `ref`, it is a numpy 1d-array
            that gives the value of the coefficient **a** in the reference prior. When set to `None`, it defaults to ``np.array([1.6,0.3])`` for gamma or 
            inverse gamma priors. When set to the reference prior, it defaults to ``np.array([0.2])``. Defaults to `None`.
        bds (ndarray, optional): a numpy 1d-array of length two that gives the lower and upper bounds of the lengthscales. Default to `None`.
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
        Rinv_y (ndarray): a numpy 1d-array that stores the product of correlation matrix inverse and the output Y. Defaults to `None`.
        vecch (bool): indicates weather the Vecchia apprxoimation is used. Defaults to `None`.
        D (int): the dimension of input data to the GP node. Defaults to `None`.
        ord (ndarray): a 1d-array that gives the ordering of input for the Vecchia approximation. Defaults to `None`.
        rev_ord (ndarray): a 1d-array that reconstructs the ordering of input from the ordered one for the Vecchia approximation. Defaults to `None`.
        m (int): the number of conditioning points in Vecchia approximation. Defaults to `None`.
        NNarray (ndarray): a 2d-array that gives the m NN for each data point after ordering for the Vecchia approximation. Defaults to `None`.
        R2 (ndarray): a 2d-array that stores the R2 of the linear regression between **global_input** and **input**. Defaults to `None`.
    """

    def __init__(self, length, scale=1., nugget=1e-6, name='sexp', prior_name='ga', prior_coef=None, bds=None, nugget_est=False, scale_est=False, input_dim=None, connect=None):
        self.type='gp'
        self.length=length
        self.scale=np.atleast_1d(scale)
        self.nugget=np.atleast_1d(nugget)
        self.name=name
        self.prior_name=prior_name
        if self.prior_name=='ga':
            if prior_coef is None:
                self.prior_coef=np.array([1.6,0.3])
            else:
                self.prior_coef=prior_coef
            self.prior_coef[0] -= 1
        elif self.prior_name=='inv_ga':
            if prior_coef is None:
                self.prior_coef=np.array([1.6,0.3])
            else:
                self.prior_coef=prior_coef
            self.prior_coef[0] += 1
        elif self.prior_name=='ref':
            if prior_coef is None:
                self.prior_coef=np.array([0.2])
            else:
                self.prior_coef=prior_coef
            self.cl=None
        self.nugget_est=nugget_est
        self.scale_est=scale_est
        self.input_dim=input_dim
        self.connect=connect
        self.para_path=None
        self.global_input=None
        self.input=None
        self.output=None
        self.rep=None
        self.rep_hetero=None
        self.Rinv=None
        self.Rinv_y=None
        self.R2sexp=None
        self.Psexp=None
        self.vecch=None
        self.D=None
        self.ord=None
        self.rev_ord=None
        self.m=None
        self.pred_m=None
        self.NNarray=None
        self.max_rep=None
        self.imp_NNarray=None
        self.imp_pointer_row=None
        self.imp_pointer_col=None
        self.nn_method='exact'
        self.ord_fun=None
        self.iter_count=0
        self.target='dgp'
        self.bds=bds
        self.R2=None
        self.loo_state=False

    def __setstate__(self, state):
        if 'g' in state:
            del state['g']
        if 'gfod' in state:
            del state['gfod']
            if state['prior_name']=='ga':
                state['prior_coef'][0] -= 1
            elif state['prior_name']=='inv_ga':
                state['prior_coef'][0] += 1
        if 'rff' in state:
            del state['rff']
        if 'vecch' not in state:
            state['vecch'] = False
        if 'M' in state:
            del state['M']
        if 'W' in state:
            del state['W']
        if 'b' in state:
            del state['b']
        if 'ord' not in state:
            state['ord'] = None
        if 'rev_ord' not in state:
            state['rev_ord'] = None
        if 'm' not in state:
            state['m'] = 25
        if 'pred_m' not in state:
            state['pred_m'] = None
        if 'NNarray' not in state:
            state['NNarray'] = None
        if 'max_rep' not in state:
            state['max_rep'] = None
        if 'rep_hetero' not in state:
            state['rep_hetero'] = None    
        if 'imp_NNarray' not in state:
            state['imp_NNarray'] = None
        if 'imp_pointer_row' not in state:
            state['imp_pointer_row'] = None
        if 'imp_pointer_col' not in state:
            state['imp_pointer_col'] = None
        if 'nn_method' not in state:
            state['nn_method'] = 'exact'
        if 'ord_fun' not in state:
            state['ord_fun'] = None
        if 'iter_count' not in state:
            state['iter_count'] = 0
        if 'target' not in state:
            state['target'] = 'dgp'
        new_R2_added = False
        if 'R2' not in state:
            state['R2'] = None
            new_R2_added = True
        if 'loo_state' not in state:
            state['loo_state'] = False
        self.__dict__.update(state)
        if new_R2_added:
            self.r2(overwritten=True)

    def compute_cl(self):
        if len(self.length)==1:
            if self.global_input is not None:
                X=np.concatenate((self.input, self.global_input),1)
            else:
                X=self.input
            if self.vecch:
                input_range = np.max(X,axis=0)-np.min(X,axis=0)
                dists = np.sqrt(np.dot(input_range, input_range))
                self.cl = dists/len(self.output)
            else:
                dists = pdist(X, metric="euclidean")
                self.cl=np.max(dists)/len(self.output)
        else:
            input_range = np.max(self.input,axis=0)-np.min(self.input,axis=0)
            if self.global_input is not None:
                g_input_range = np.max(self.global_input,axis=0)-np.min(self.global_input,axis=0)
                input_range = np.concatenate((input_range, g_input_range))
            self.cl=input_range/len(self.output)**(1/len(self.length))

    def r2(self, overwritten = False):
        """Compute R2 of the linear regression between **global_input** and **input**.
        """
        if self.global_input is not None:
            X = np.concatenate((self.global_input, np.ones((len(self.global_input),1))), axis=1)
            if matrix_rank(self.global_input) == matrix_rank(X):
                X = self.global_input
            _, resids = lstsq(X, self.input, rcond = None)[:2]
            rsq = 1 - resids / (len(self.input) * np.var(self.input, axis=0))
            if overwritten:
                self.R2 = np.atleast_2d(rsq)
            else:
                self.R2 = np.vstack((self.R2,rsq))

    def ord_nn(self, ord = None, NNarray = None, pointer=False):
        """Specify the ordering and NN for the Vecchia approximation
        """
        if ord is None:
            if self.ord_fun is None:
                self.ord = np.random.permutation(self.input.shape[0])
            else:
                if self.global_input is not None:
                    X = np.concatenate((self.input, self.global_input),1)/self.length
                else:
                    X = self.input/self.length
                self.ord = self.ord_fun(X)
        else:
            self.ord = ord
        self.rev_ord = np.argsort(self.ord)
        if NNarray is None:
            if self.global_input is not None:
                X = np.concatenate((self.input, self.global_input),1)/self.length
            else:
                X = self.input/self.length
            self.NNarray = nn(X[self.ord], self.m, method = self.nn_method)
        else:
            self.NNarray = NNarray
        if pointer:
            NNs = get_pred_nn(X[self.ord], X[self.ord], self.m)[:,1::]
            n = X.shape[0]
            prev = NNs < np.tile(np.arange(n), (self.m-1, 1)).T
            NNs[prev] = NNs[prev] + n
            self.imp_NNarray = np.hstack((np.arange(n).reshape(-1,1) + n, np.arange(n).reshape(-1,1), NNs))
            if self.max_rep is None:
                self.imp_pointer_row, self.imp_pointer_col = imp_pointers(self.imp_NNarray)
            else:
                self.imp_pointer_row, self.imp_pointer_col = imp_pointers_rep(self.imp_NNarray, self.max_rep, self.rep_hetero, self.ord)

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
        """Update the model parameters (lengthscales and nugget).

        Args:
            log_theta (ndarray): optimised numpy 1d-array of log-transformed lengthscales and nugget.
        """
        theta=np.exp(log_theta)
        if self.nugget_est:
            self.length=theta[0:-1]
            self.nugget=theta[[-1]]
        else:
            self.length=theta
            
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
        #with np.errstate(divide='ignore'):
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
        
    def gfod(self, x):
        if self.prior_name=='ga':
            return self.prior_coef[0]-self.prior_coef[1]*x
        else:
            -self.prior_coef[0]+self.prior_coef[1]/x
    
    def log_prior(self):
        """Compute the value of log priors specified to the lengthscales and nugget. 

        Returns:
            ndarray: a numpy 1d-array giving the sum of log priors of the lengthscales and nugget. 
        """
        if self.prior_name=='ref':
            a, b=self.prior_coef[0], self.prior_coef[1]
            t=np.sum(self.cl/self.length)+self.nugget
            lp=a*np.log(t)-b*t
        else:
            lp=g(self.prior_coef[0], self.prior_coef[1], self.length, self.prior_name)
            if self.nugget_est:
                lp+=g(self.prior_coef[0], self.prior_coef[1], self.nugget, self.prior_name)
        return lp

    def log_prior_fod(self):
        """Compute the first order derivatives of log priors wrt the log-transformed lengthscales and nugget.

        Returns:
            ndarray: a numpy 1d-array (whose length equal to the total number of lengthscales and nugget)
            giving the first order derivatives of log priors wrt the log-transformed lengthscales and nugget.
        """
        if self.prior_name=='ref':
            a, b=self.prior_coef[0], self.prior_coef[1]
            t=np.sum(self.cl/self.length)+self.nugget
            fod=(b-a/t)*self.cl/self.length
            if self.nugget_est:
                fod_nugget=(a/t-b)*self.nugget
                fod=np.concatenate((fod,fod_nugget))
        else:  
            fod=self.gfod(self.length)
            if self.nugget_est:
                fod=np.concatenate((fod, self.gfod(self.nugget)))
        return fod
    
    def llik(self,x):
        """Compute the negative log-likelihood function of the GP and the first order derivatives of the negative log-likelihood function wrt log-transformed model parameters..

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            tuple: a tuple is returned. The tuple contains two numpy 1d-arrays. The first one gives the negative log-likelihood. The second one (whose length equal to the total number of lengthscales and nugget)
            contains first order derivatives of the negative log-likelihood function wrt log-transformed lengthscales and nugget.
        """
        self.update(x)
        n=len(self.output)
        K,Kt=self.k_matrix(fod_eval=True)
        L=cholesky(K,lower=True,check_finite=False)
        KinvKt=np.array([cho_solve((L, True), Kt_i, check_finite=False) for Kt_i in Kt])
        #tr_KinvKt=np.trace(KinvKt, axis1=1, axis2=2)
        #logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        tr_KinvKt=trace_nb(KinvKt)
        logdet=logdet_nb(L)
        KinvY=cho_solve((L, True), self.output, check_finite=False)
        YKinvKtKinvY=((self.output).T@KinvKt@KinvY).flatten()
        YKinvY=(self.output).T@KinvY
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if self.scale_est:
            self.scale=(YKinvY/n).flatten()
            neg_llik=0.5*(logdet+n*np.log(self.scale))
            neg_St=-P1-P2/self.scale
        else:
            neg_llik=0.5*(logdet+YKinvY/self.scale) 
            neg_St=-P1-P2/self.scale
        neg_llik=neg_llik.flatten()
        if self.prior_name is not None:
            neg_llik=neg_llik-self.log_prior()
            neg_St=neg_St-self.log_prior_fod()
        return neg_llik, neg_St
    
    def llik_vecch(self,x):
        """Compute the negative log-likelihood function of the GP under Vecchia approximation.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            tuple: a tuple is returned. The tuple contains two numpy 1d-arrays. The first one gives the negative log-likelihood. The second one (whose length equal to the total number of lengthscales and nugget)
            contains first order derivatives of the negative log-likelihood function wrt log-transformed lengthscales and nugget.
        """
        self.update(x)
        if self.connect is not None:
            X = np.concatenate((self.input,self.global_input),1)
        else:
            X = self.input
        neg_llik, neg_St, self.scale = vecchia_nllik(X[self.ord], self.output[self.ord], self.NNarray, self.scale[0], self.length, self.nugget[0], self.name, self.scale_est, self.nugget_est)
        if self.prior_name is not None:
            neg_llik=neg_llik-self.log_prior()
            neg_St=neg_St-self.log_prior_fod()
        return neg_llik, neg_St

    def log_likelihood_func(self):
        cov=self.scale*self.k_matrix()
        L=cholesky(cov, lower=True, check_finite=False)
        #L=np.linalg.cholesky(cov)
        #logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        logdet=logdet_nb(L)
        quad=(self.output).T@cho_solve((L, True), self.output, check_finite=False)
        llik=-0.5*(logdet+quad)
        if self.prior_name=='ref':
            self.compute_cl()
            llik+=self.log_prior()
        return llik

    def log_likelihood_func_vecch(self):
        """Compute Gaussian log-likelihood function using the Vecchia approximation.
        """
        if self.connect is not None:
            X=np.concatenate((self.input,self.global_input),1)
        else:
            X=self.input
        llik = vecchia_llik(X[self.ord], self.output[self.ord], self.NNarray, self.scale[0], self.length, self.nugget[0], self.name)
        if self.prior_name=='ref':
            self.compute_cl()
            llik+=self.log_prior()
        return llik
    
    def callback(self, xk):
        self.iter_count += 1
        if self.iter_count & (self.iter_count-1) == 0:
            self.ord_nn()

    def maximise(self, method='L-BFGS-B'):
        """Optimise and update model parameters by minimising the negative log-likelihood function.

        Args:
            method (str, optional): optimisation algorithm. Defaults to `L-BFGS-B`.
        """
        initial_theta_trans=self.log_t()
        if self.nugget_est:
            if self.bds is None:
                lb=np.concatenate((-np.inf*np.ones(len(initial_theta_trans)-1),np.log([1e-8])))
                if self.prior_name=='ref':
                    ub=np.concatenate((13.*np.ones(len(initial_theta_trans)-1), [np.inf]))
                else:
                    ub=np.inf*np.ones(len(initial_theta_trans))
            else:
                with np.errstate(divide='ignore'):
                    lb=np.concatenate((np.log(self.bds[0])*np.ones(len(initial_theta_trans)-1),np.log([1e-8])))
                ub=np.concatenate((np.log(self.bds[1])*np.ones(len(initial_theta_trans)-1),[np.inf]))
            bd=Bounds(lb, ub)
            if self.vecch:
                if self.target=='gp' and len(self.length)!=1:
                    _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                    self.iter_count = 0
                else:
                    _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
            else:
                _ = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
        else:
            if self.bds is None:
                if self.prior_name=='ref':
                    lb=-np.inf*np.ones(len(initial_theta_trans))
                    ub=13.*np.ones(len(initial_theta_trans))
                    bd=Bounds(lb, ub)
                    if self.vecch:
                        if self.target=='gp' and len(self.length)!=1:
                            _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                            self.iter_count = 0
                        else:
                            _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                    else:
                        _ = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                else:
                    if self.vecch:
                        if self.target=='gp' and len(self.length)!=1:
                            _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                            self.iter_count = 0                       
                        else:
                            _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                    else:
                        _ = minimize(self.llik, initial_theta_trans, method=method, jac=True, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
            else:
                with np.errstate(divide='ignore'):
                    lb=np.log(self.bds[0])*np.ones(len(initial_theta_trans))
                ub=np.log(self.bds[1])*np.ones(len(initial_theta_trans))
                bd=Bounds(lb, ub)
                if self.vecch:
                    if self.target=='gp' and len(self.length)!=1:
                        _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                        self.iter_count = 0
                    else:
                        _ = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                else:
                    _ = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
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
        num_x, num_thread = x.shape[0], get_thread()
        parallel = True if num_x > num_thread else False
        if self.vecch:
            if z is not None:
                x=np.concatenate((x, z),1)
                w=np.concatenate((self.input, self.global_input),1)
            else:
                w = self.input
            NNarray = get_pred_nn(x/self.length, w/self.length, self.pred_m, method = self.nn_method)
            if self.loo_state:
                NNarray = NNarray[:,1:]
            if parallel:
                m,v = gp_vecch(x,w,NNarray,self.output,self.scale[0],self.length,self.nugget[0],self.name)
            else:
                m,v = gp_vecch_non_parallel(x,w,NNarray,self.output,self.scale[0],self.length,self.nugget[0],self.name)
        else:
            if parallel:
                m,v=gp(x,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.scale,self.length,self.nugget,self.name)
            else:
                m,v=gp_non_parallel(x,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.scale,self.length,self.nugget,self.name)
        return m,v

    def linkgp_prediction(self,m,v,z):
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

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions (that are 
            represented by predictive means and variances).
        """
        num_x, num_thread = m.shape[0], get_thread()
        parallel = True if num_x > num_thread else False
        if self.vecch:
            if z is not None:
                x = np.concatenate((m, z),1)
                w = np.concatenate((self.input, self.global_input),1)
            else:
                x = m
                w = self.input
            NNarray = get_pred_nn(x/self.length, w/self.length, self.pred_m, method = self.nn_method)
            if self.loo_state:
                NNarray = NNarray[:,1:]
            if parallel:
                m,v = link_gp_vecch(m, v, z, self.input, self.global_input, NNarray, self.output, self.scale[0], self.length, self.nugget[0], self.name)
            else:
                m,v = link_gp_vecch_non_parallel(m, v, z, self.input, self.global_input, NNarray, self.output, self.scale[0], self.length, self.nugget[0], self.name)
        else:
            if parallel:
                m,v=link_gp(m,v,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.R2sexp,self.Psexp,self.scale[0],self.length,self.nugget[0],self.name)
            else:
                m,v=link_gp_non_parallel(m,v,z,self.input,self.global_input,self.Rinv,self.Rinv_y,self.R2sexp,self.Psexp,self.scale[0],self.length,self.nugget[0],self.name)
        return m,v

    def linkgp_prediction_full(self,m,v,m_z,v_z,z):
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

        Returns:
            tuple: a tuple of two 1d-arrays giving the means and variances at the testing input data positions (that are 
            represented by predictive means and variances).
        """
        num_x, num_thread = m.shape[0], get_thread()
        parallel = True if num_x > num_thread else False
        m=np.concatenate((m,m_z),axis=1)
        v=np.concatenate((v,v_z),axis=1)
        idx1=np.arange(np.shape(m_z)[1])
        idx2=np.arange(np.shape(m_z)[1],np.shape(self.global_input)[1])
        overall_input=np.concatenate((self.input,self.global_input[:,idx1]),axis=1)
        if self.vecch:
            if z is not None:
                x = np.concatenate((m, z),1)
                w = np.concatenate((self.input, self.global_input),1)
            else:
                x = m
                w = overall_input
            NNarray = get_pred_nn(x/self.length, w/self.length, self.pred_m, method = self.nn_method)
            if parallel:     
                m,v = link_gp_vecch(m, v, z, overall_input, self.global_input[:,idx2], NNarray, self.output, self.scale[0], self.length, self.nugget[0], self.name)
            else:
                m,v = link_gp_vecch_non_parallel(m, v, z, overall_input, self.global_input[:,idx2], NNarray, self.output, self.scale[0], self.length, self.nugget[0], self.name)
        else:
            if self.name=='sexp':
                if len(self.length)==1:
                    global_input_l=self.global_input[:,idx1]/self.length
                else:
                    D=np.shape(self.input)[1]
                    global_input_l=self.global_input[:,idx1]/(self.length[D::][idx1])
                dists = pdist(global_input_l, metric="sqeuclidean")
                R2sexp_global = squareform(np.exp(-dists/2))
                np.fill_diagonal(R2sexp_global, 1)
                R2sexp = self.R2sexp*R2sexp_global
                Psexp_global = Pmatrix(global_input_l)
                Psexp = np.concatenate((self.Psexp,Psexp_global),axis=0)
            else:
                R2sexp, Psexp = self.R2sexp, self.Psexp
            if parallel:
                m,v=link_gp(m,v,z,overall_input,self.global_input[:,idx2],self.Rinv,self.Rinv_y,R2sexp,Psexp,self.scale[0],self.length,self.nugget[0],self.name)
            else:
                m,v=link_gp_non_parallel(m,v,z,overall_input,self.global_input[:,idx2],self.Rinv,self.Rinv_y,R2sexp,Psexp,self.scale[0],self.length,self.nugget[0],self.name)
        return m,v

    def compute_stats(self):
        """Compute and store key statistics for the GP predictions
        """
        R=self.k_matrix()
        #U, s, Vh = np.linalg.svd(R)
        #self.Rinv=Vh.T@np.diag(s**-1)@U.T
        #L=np.linalg.cholesky(R)
        #self.Rinv_y=cho_solve((L, True), self.output, check_finite=False)
        #self.Rinv=pinvh(R,check_finite=False)
        #self.Rinv_y=np.dot(self.Rinv,self.output).flatten()
        try:
            L=np.linalg.cholesky(R)
            self.Rinv=cho_solve((L, True), np.eye(len(R)), check_finite=False)
            self.Rinv_y=cho_solve((L, True), self.output, check_finite=False).flatten()
        except LinAlgError:
            self.Rinv=pinvh(R,check_finite=False)
            self.Rinv_y=np.dot(self.Rinv,self.output).flatten()
        if self.name=='sexp':
            if self.global_input is None:
                X_l=self.input/self.length
            else:
                if len(self.length)==1:
                    X_l=self.input/self.length
                else:
                    D=np.shape(self.input)[1]
                    X_l=self.input/self.length[:D]
            dists = pdist(X_l, metric="sqeuclidean")
            self.R2sexp = squareform(np.exp(-dists/2))
            np.fill_diagonal(self.R2sexp, 1)
            self.Psexp = Pmatrix(X_l)

def combine(*layers):
    """Combine layers into one list as a DGP or linked (D)GP structure.

    Args:
        layers (list): a sequence of lists, each of which contains the GP nodes (defined by the :class:`.kernel` class), 
            likelihood nodes (e.g., defined by the :class:`.Poisson` class), or containers (defined by the :class:`.container` class)
            in that layer.

    Returns:
        list: a list of layers defining the DGP or linked (D)GP structure.
    """
    all_layer=[]
    for layer in layers:
        all_layer.append(layer)
    return all_layer