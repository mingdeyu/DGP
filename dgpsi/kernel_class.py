import numpy as np
from numpy.linalg import LinAlgError, lstsq#, matrix_rank
from scipy.optimize import minimize, Bounds
from scipy.linalg import cho_solve, pinvh, cholesky
from scipy.spatial.distance import pdist
from .functions import gp, gp_non_parallel, logdet_nb, g, link_gp_matern25_noz_serial, link_gp_matern25_noz_parallel, link_gp_matern25_withz_serial, link_gp_matern25_withz_parallel, link_gp_sexp_noz_serial, link_gp_sexp_noz_parallel, link_gp_sexp_withz_serial, link_gp_sexp_withz_parallel
from .vecchia import nn, vecchia_llik, vecchia_nllik, get_pred_nn, gp_vecch, gp_vecch_non_parallel, imp_pointers, dK_matrix_nb, K_matrix_nb, preprocess_nn, link_gp_vecch_sexp_noz_parallel, link_gp_vecch_sexp_noz_serial, link_gp_vecch_sexp_withz_parallel, link_gp_vecch_sexp_withz_serial, link_gp_vecch_matern25_noz_parallel, link_gp_vecch_matern25_noz_serial, link_gp_vecch_matern25_withz_parallel, link_gp_vecch_matern25_withz_serial
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
        self.sum_residual=None
        self.W_diag=None
        self.origin_n=None

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
        if 'sum_residual' not in state:
            state['sum_residual'] = None
        if 'W_diag' not in state:
            state['W_diag'] = None
        if 'NN_rev' not in state:
            state['NN_rev'] = None
        if 'NN_count' not in state:
            state['NN_count'] = None
        if 'origin_n' not in state:
            state['origin_n'] = None
        self.__dict__.update(state)
        if new_R2_added:
            self.r2(overwritten=True)
        if self.output is not None and self.W_diag is None:
            self.W_diag = np.ones(self.output.shape[0], dtype=np.float64)
        if self.output is not None and self.sum_residual is None:
            self.sum_residual = -1.0
        if self.NNarray is not None and (self.NN_rev is None or self.NN_count is None):
            NN = np.ascontiguousarray(self.NNarray.astype(np.int32))
            self.NN_rev, self.NN_count = preprocess_nn(NN)
        if self.origin_n is None:
            if self.rep is not None:
                self.origin_n = len(self.rep)
            elif self.output is not None:
                self.origin_n = len(self.output)

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
        if self.global_input is None:
            return

        G = self.global_input
        Y = self.input
        N = G.shape[0]

        X = np.concatenate((G, np.ones((N, 1), dtype=G.dtype)), axis=1)

        beta, resids, _, _ = lstsq(X, Y, rcond=None)

        if resids.size == 0:
            E = Y - X @ beta
            resids = np.sum(E * E, axis=0)

        rsq = 1.0 - resids / (N * np.var(Y, axis=0)) 

        if overwritten or self.R2 is None:
            self.R2 = np.atleast_2d(rsq)
        else:
            self.R2 = np.vstack((self.R2, rsq))

    # def ord_nn(self, ord = None, NNarray = None, pointer=False):
    #     """Specify the ordering and NN for the Vecchia approximation
    #     """
    #     if ord is None:
    #         if self.ord_fun is None:
    #             self.ord = np.random.permutation(self.input.shape[0])
    #         else:
    #             if self.global_input is not None:
    #                 X = np.concatenate((self.input, self.global_input),1)/self.length
    #             else:
    #                 X = self.input/self.length
    #             self.ord = self.ord_fun(X)
    #     else:
    #         self.ord = ord
    #     self.rev_ord = np.argsort(self.ord)
    #     if NNarray is None:
    #         if self.global_input is not None:
    #             X = np.concatenate((self.input, self.global_input),1)/self.length
    #         else:
    #             X = self.input/self.length
    #         self.NNarray = nn(X[self.ord], self.m, method = self.nn_method)
    #     else:
    #         self.NNarray = NNarray
    #     NN = np.ascontiguousarray(self.NNarray.astype(np.int32))
    #     self.NN_rev, self.NN_count = preprocess_nn(NN)
    #     if pointer:
    #         NNs = get_pred_nn(X[self.ord], X[self.ord], self.m)[:,1::]
    #         n = X.shape[0]
    #         prev = NNs < np.tile(np.arange(n), (self.m-1, 1)).T
    #         NNs[prev] = NNs[prev] + n
    #         self.imp_NNarray = np.hstack((np.arange(n).reshape(-1,1) + n, np.arange(n).reshape(-1,1), NNs))
    #         self.imp_pointer_row, self.imp_pointer_col = imp_pointers(self.imp_NNarray)

    def ord_nn(self, ord=None, NNarray=None, pointer=False, rev_ord=None, NN_rev=None, NN_count=None):
        """Specify the ordering and NN for the Vecchia approximation.

        Reuse logic:
        - If `ord` is provided, assign it; otherwise compute (random or via ord_fun).
        - If `rev_ord` is provided, assign it; otherwise compute argsort(ord).
        - If `NNarray` is provided, assign it; otherwise compute via nn(X[ord], m, method).
        - If `NN_rev/NN_count` are provided, assign them; otherwise compute via preprocess_nn.
        - If pointer=True, compute pointer arrays for THIS kernel (even if ord/NN were reused).
        """

        # ----------------------------
        # Decide if we need scaled X
        # ----------------------------
        need_X_for_ord = (ord is None and self.ord_fun is not None)
        need_X_for_nn = (NNarray is None)
        need_X_for_ptr = pointer

        X = None
        if need_X_for_ord or need_X_for_nn or need_X_for_ptr:
            if self.global_input is not None:
                X = np.concatenate((self.input, self.global_input), 1) / self.length
            else:
                X = self.input / self.length

        # ----------
        # Ordering
        # ----------
        if ord is None:
            if self.ord_fun is None:
                self.ord = np.random.permutation(self.input.shape[0])
            else:
                self.ord = self.ord_fun(X)
        else:
            self.ord = ord

        # ----------
        # rev_ord
        # ----------
        if rev_ord is None:
            self.rev_ord = np.argsort(self.ord)
        else:
            self.rev_ord = rev_ord

        # ----------
        # NNarray
        # ----------
        if NNarray is None:
            X_ord = X[self.ord]
            NN = nn(X_ord, self.m, method=self.nn_method)
            self.NNarray = np.ascontiguousarray(NN, dtype=np.int32)
        else:
            self.NNarray = np.ascontiguousarray(NNarray, dtype=np.int32)

        # ----------
        # NN_rev / NN_count
        # ----------
        if NN_rev is None or NN_count is None:
            # preprocess_nn expects contiguous int32
            self.NN_rev, self.NN_count = preprocess_nn(self.NNarray)
        else:
            self.NN_rev = NN_rev
            self.NN_count = NN_count

        # ----------
        # Pointer (per kernel!)
        # ----------
        if pointer:
            # X is guaranteed to exist because need_X_for_ptr -> True above
            X_ord = X[self.ord]

            NNs = get_pred_nn(X_ord, X_ord, self.m)[:, 1:]   # (n, m-1)
            n = X_ord.shape[0]

            prev = NNs < np.arange(n)[:, None]
            NNs[prev] += n

            imp = np.empty((n, self.m + 1), dtype=np.int32)  # 2 + (m-1) columns
            base = np.arange(n, dtype=np.int32)
            imp[:, 0] = base + n
            imp[:, 1] = base
            imp[:, 2:] = NNs

            self.imp_NNarray = imp
            self.imp_pointer_row, self.imp_pointer_col = imp_pointers(self.imp_NNarray)

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
        if self.connect is not None:
            X = np.concatenate((self.input,self.global_input),1)
        else:
            X = self.input

        n = self.output.shape[0]

        nuggeti = self.nugget[0] * self.W_diag

        if fod_eval:
            return dK_matrix_nb(X, self.length, nuggeti, self.name, self.nugget_est, (n>=400))
        else:
            return K_matrix_nb(X, self.length, nuggeti, self.name, (n>=400))
        
    def gfod(self, x):
        if self.prior_name=='ga':
            return self.prior_coef[0]-self.prior_coef[1]*x
        else:
            return -self.prior_coef[0]+self.prior_coef[1]/x
    
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
    
    def llik(self, x):
        """Compute the negative log-likelihood function of the GP and the first order derivatives of the negative log-likelihood function wrt log-transformed model parameters..

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            tuple: a tuple is returned. The tuple contains two numpy 1d-arrays. The first one gives the negative log-likelihood. The second one (whose length equal to the total number of lengthscales and nugget)
            contains first order derivatives of the negative log-likelihood function wrt log-transformed lengthscales and nugget.
        """
        self.update(x)

        y = self.output[:, 0]
        n = y.shape[0]

        K, Kt = self.k_matrix(fod_eval=True)

        p = Kt.shape[0]
        L = cholesky(K, lower=True, check_finite=False)
        alpha = cho_solve((L, True), y, check_finite=False)   # (n,1)

        logdet = logdet_nb(L)
        yKy = np.atleast_1d(y @ alpha)

        if self.scale_est:
            if self.rep is None:
                self.scale = yKy / n
                neg_llik = 0.5 * (logdet + n * np.log(self.scale))
            else:
                m = len(self.rep)
                self.scale = (yKy + self.sum_residual / self.nugget) / m
                neg_llik = 0.5 * (logdet + m * np.log(self.scale))
        else:
            neg_llik = 0.5 * (logdet + yKy / self.scale)

        Kinv = cho_solve((L, True), np.eye(n, dtype=K.dtype), check_finite=False)

        a = alpha.ravel()
        W = Kinv - np.outer(a, a) / self.scale

        grad = 0.5 * (Kt.reshape(p, -1) @ W.ravel())

        if self.rep is not None and self.nugget_est:
            m = len(self.rep)
            if self.scale_est:
                neg_llik += 0.5 * (m - n) * np.log(self.nugget)
            else:
                neg_llik += 0.5 * (self.sum_residual / (self.scale * self.nugget) + (m - n) * np.log(self.nugget))

            grad[-1] += 0.5 * (-self.sum_residual / (self.scale * self.nugget) + (m - n))

        neg_llik = np.asarray(neg_llik)

        if self.prior_name is not None:
            neg_llik = neg_llik - self.log_prior()
            grad = grad - self.log_prior_fod()

        return neg_llik, grad
    
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
        nugget_diag = self.W_diag
        ord = self.ord
        X0, y0, nugget_diag0 = X[ord], self.output[ord,0], nugget_diag[ord]
        neg_llik, neg_St, self.scale = vecchia_nllik(X0, y0, self.NN_rev, self.NN_count, self.scale[0], self.length, self.nugget[0], nugget_diag0, self.name, self.scale_est, self.nugget_est, self.origin_n, self.sum_residual)
        if self.prior_name is not None:
            neg_llik=neg_llik-self.log_prior()
            neg_St=neg_St-self.log_prior_fod()
        return neg_llik, neg_St

    def log_likelihood_func(self):
        y = self.output.ravel()
        K = self.k_matrix()
        L = cholesky(K, lower=True, check_finite=False)
        #L=np.linalg.cholesky(cov)
        #logdet=2*np.sum(np.log(np.abs(np.diag(L))))
        logdet = logdet_nb(L)
        alpha = cho_solve((L, True), y, check_finite=False)
        quad = y @ alpha
        s = self.scale[0]  # shape (1,)
        llik = -0.5 * (logdet + quad / s)
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
        nugget_diag = self.W_diag
        ord = self.ord
        X0, y0, nugget_diag0 = X[ord], self.output[ord,0], nugget_diag[ord]
        llik = vecchia_llik(X0, y0, self.NN_rev, self.NN_count, self.scale[0], self.length, self.nugget[0], nugget_diag0, self.name)
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
                    res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                    self.iter_count = 0
                else:
                    res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
            else:
                res = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
        else:
            if self.bds is None:
                if self.prior_name=='ref':
                    lb=-np.inf*np.ones(len(initial_theta_trans))
                    ub=13.*np.ones(len(initial_theta_trans))
                    bd=Bounds(lb, ub)
                    if self.vecch:
                        if self.target=='gp' and len(self.length)!=1:
                            res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                            self.iter_count = 0
                        else:
                            res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                    else:
                        res = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                else:
                    if self.vecch:
                        if self.target=='gp' and len(self.length)!=1:
                            res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                            self.iter_count = 0                       
                        else:
                            res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                    else:
                        res = minimize(self.llik, initial_theta_trans, method=method, jac=True, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
            else:
                with np.errstate(divide='ignore'):
                    lb=np.log(self.bds[0])*np.ones(len(initial_theta_trans))
                ub=np.log(self.bds[1])*np.ones(len(initial_theta_trans))
                bd=Bounds(lb, ub)
                if self.vecch:
                    if self.target=='gp' and len(self.length)!=1:
                        res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, callback=self.callback, options={'maxfun': np.max((50,20+5*self.D))})
                        self.iter_count = 0
                    else:
                        res = minimize(self.llik_vecch, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
                else:
                    res = minimize(self.llik, initial_theta_trans, method=method, jac=True, bounds=bd, options={'maxiter': 100, 'maxfun': np.max((30,20+5*self.D))})
        self.update(res.x)
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
            nugget_diag = self.W_diag
            if parallel:
                m,v = gp_vecch(x,w,NNarray,self.output,self.scale[0],self.length,self.nugget[0],nugget_diag,self.name)
            else:
                m,v = gp_vecch_non_parallel(x,w,NNarray,self.output,self.scale[0],self.length,self.nugget[0],nugget_diag,self.name)
        else:
            if z is not None:
                x = np.concatenate((x, z), 1) / self.length                      
                w = np.concatenate((self.input, self.global_input), 1) / self.length
            else:   
                x = x / self.length                                      
                w = self.input / self.length
            if parallel:
                m,v=gp(x,w,self.Rinv,self.Rinv_y,self.scale[0],self.nugget[0],self.name)
            else:
                m,v=gp_non_parallel(x,w,self.Rinv,self.Rinv_y,self.scale[0],self.nugget[0],self.name)
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
            nugget_diag = self.W_diag
            scale = self.scale[0]
            nugget = self.nugget[0]
            y = self.output
            w1 = self.input

            if self.name == "sexp":
                Dw = w1.shape[1]
                if z is None:
                    length_w = np.full(Dw, self.length[0], dtype=np.float64) if len(self.length) == 1 else self.length
                    inv_len_w = 1.0 / length_w

                    if parallel:
                        return link_gp_vecch_sexp_noz_parallel(m, v, w1, NNarray, y, scale, length_w, inv_len_w, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_sexp_noz_serial(m, v, w1, NNarray, y, scale, length_w, inv_len_w, nugget, nugget_diag)
                else:
                    Dz = z.shape[1]
                    length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64) if len(self.length) == 1 else self.length
                    length_w = length_full[:Dw]
                    length_z = length_full[Dw:Dw + Dz]
                    inv_len_w = 1.0 / length_w
                    inv_len_z = 1.0 / length_z

                    if parallel:
                        return link_gp_vecch_sexp_withz_parallel(m, v, z, w1, self.global_input, NNarray, y, scale, length_full, inv_len_w, inv_len_z, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_sexp_withz_serial(m, v, z, w1, self.global_input, NNarray, y, scale, length_full, inv_len_w, inv_len_z, nugget, nugget_diag)

            # ===== matern2.5 =====
            Dw = w1.shape[1]
            if z is None:
                length_w = np.full(Dw, self.length[0], dtype=np.float64) if len(self.length) == 1 else self.length

                if parallel:
                    return link_gp_vecch_matern25_noz_parallel(m, v, w1, NNarray, y, scale, length_w, nugget, nugget_diag)
                else:
                    return link_gp_vecch_matern25_noz_serial(m, v, w1, NNarray, y, scale, length_w, nugget, nugget_diag)
            else:
                Dz = z.shape[1]
                length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64) if len(self.length) == 1 else self.length
                length_w = length_full[:Dw]
                length_z = length_full[Dw:Dw + Dz]
                inv_len_z = 1.0 / length_z

                if parallel:
                    return link_gp_vecch_matern25_withz_parallel(m, v, z, w1, self.global_input, NNarray, y, scale, length_full, length_w, inv_len_z, nugget, nugget_diag)
                else:
                    return link_gp_vecch_matern25_withz_serial(m, v, z, w1, self.global_input, NNarray, y, scale, length_full, length_w, inv_len_z, nugget, nugget_diag)
        else:
            w1 = self.input
            Rinv = self.Rinv
            Rinv_y = self.Rinv_y
            scale = self.scale[0]
            nugget = self.nugget[0]
            if self.name == "sexp":
                Dw = w1.shape[1]
                if z is None:
                    if len(self.length) == 1:
                        length_w = np.full(Dw, self.length[0], dtype=np.float64)
                    else:
                        length_w = self.length

                    inv_len_w = 1.0 / length_w

                    if parallel:
                        m2, v2 = link_gp_sexp_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget) 
                    else:
                        m2, v2 = link_gp_sexp_noz_serial(m, v, w1, Rinv, Rinv_y, scale, inv_len_w, nugget)
                    return m2, v2

                else:
                    Dz = z.shape[1]
                    if len(self.length) == 1:
                        length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64)
                    else:
                        length_full = self.length

                    length_w = length_full[:Dw]
                    length_z = length_full[Dw:Dw + Dz]
                    inv_len_w = 1.0 / length_w
                    inv_len_z = 1.0 / length_z

                    if parallel:
                        m2, v2 = link_gp_sexp_withz_parallel(m, v, z, self.input, self.global_input, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget)  # CHANGED
                    else:
                        m2, v2 = link_gp_sexp_withz_serial(m, v, z, self.input, self.global_input, Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget)    # CHANGED
                    return m2, v2
            else:
                Dw = w1.shape[1]
                if z is None:
                    if len(self.length) == 1:
                        length_w = np.full(Dw, self.length[0], dtype=np.float64)
                    else:
                        length_w = self.length

                    if parallel:
                        m2, v2 = link_gp_matern25_noz_parallel(m, v, w1, Rinv, Rinv_y, scale, length_w, nugget)
                    else:
                        m2, v2 = link_gp_matern25_noz_serial(m, v, w1, Rinv, Rinv_y, scale, length_w, nugget)
                    return m2, v2

                else:
                    Dz = z.shape[1]
                    if len(self.length) == 1:
                        length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64)
                    else:
                        length_full = self.length

                    length_w = length_full[:Dw]
                    length_z = length_full[Dw:Dw + Dz]

                    inv_len_z = 1.0 / length_z

                    if parallel:
                        m2, v2 = link_gp_matern25_withz_parallel(m, v, z, self.input, self.global_input, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget)
                    else:
                        m2, v2 = link_gp_matern25_withz_serial(m, v, z, self.input, self.global_input, Rinv, Rinv_y, scale, length_w, inv_len_z, nugget)
                    return m2, v2

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
        k = m_z.shape[1]
        overall_input=np.concatenate((self.input,self.global_input[:,:k]),axis=1)
        if self.vecch:
            if z is not None:
                x = np.concatenate((m, z),1)
                w = np.concatenate((self.input, self.global_input),1)
            else:
                x = m
                w = overall_input
            NNarray = get_pred_nn(x/self.length, w/self.length, self.pred_m, method = self.nn_method)
            nugget_diag = self.W_diag
            scale = self.scale[0]
            nugget = self.nugget[0]

            Dw = overall_input.shape[1]
            Dz = 0 if z is None else z.shape[1]

            # ---- length expansion (match your style) ----
            if len(self.length) == 1:
                length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64)
            else:
                length_full = np.asarray(self.length, dtype=np.float64)

            if self.name == "sexp":
                # sexp needs inv_len_w, inv_len2_w, inv_len_z, and length_full for local K
                if z is None:
                    length_w = length_full
                    inv_len_w = 1.0 / length_w
                    if parallel:
                        return link_gp_vecch_sexp_noz_parallel(m, v, overall_input, NNarray, self.output, scale, length_w, inv_len_w, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_sexp_noz_serial(m, v, overall_input, NNarray, self.output, scale, length_w, inv_len_w, nugget, nugget_diag)
                else:
                    length_w = length_full[:Dw]
                    inv_len_w = 1.0 / length_w

                    length_z = length_full[Dw:Dw + Dz]
                    inv_len_z = 1.0 / length_z

                    if parallel:
                        return link_gp_vecch_sexp_withz_parallel(
                            m, v, z, overall_input, self.global_input[:, k:], NNarray, self.output, scale, length_full, inv_len_w, inv_len_z, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_sexp_withz_serial(m, v, z, overall_input, self.global_input[:, k:], NNarray, self.output, scale, length_full, inv_len_w, inv_len_z, nugget, nugget_diag)
            else:
                # matern2.5 needs (length_w, inv_len_w, inv_len2_w, len2/3/4_w) + inv_len_z + length_full
                if z is None:
                    length_w = length_full
                    if parallel:
                        return link_gp_vecch_matern25_noz_parallel(m, v, overall_input, NNarray, self.output, scale, length_w, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_matern25_noz_serial(m, v, overall_input, NNarray, self.output, scale, length_w, nugget, nugget_diag)
                else:
                    length_w = length_full[:Dw]

                    length_z = length_full[Dw:Dw + Dz]
                    inv_len_z = 1.0 / length_z

                    if parallel:
                        return link_gp_vecch_matern25_withz_parallel(m, v, z, overall_input, self.global_input[:, k:], NNarray, self.output, scale, length_full, length_w, inv_len_z, nugget, nugget_diag)
                    else:
                        return link_gp_vecch_matern25_withz_serial(m, v, z, overall_input, self.global_input[:, k:], NNarray, self.output, scale, length_full, length_w, inv_len_z, nugget, nugget_diag)
        else:
            Rinv = self.Rinv
            Rinv_y = self.Rinv_y
            scale = self.scale[0]
            nugget = self.nugget[0]

            Dw = overall_input.shape[1]
            Dz = 0 if z is None else z.shape[1]

            if len(self.length) == 1:
                length_full = np.full(Dw + Dz, self.length[0], dtype=np.float64)
            else:
                length_full = np.asarray(self.length, dtype=np.float64)

            if self.name == "sexp":
                if z is None:
                    length_w = length_full
                    inv_len_w = 1.0 / length_w

                    if parallel:
                        return link_gp_sexp_noz_parallel(m, v, overall_input, Rinv, Rinv_y, scale, inv_len_w, nugget)
                    else:
                        return link_gp_sexp_noz_serial(m, v, overall_input, Rinv, Rinv_y, scale, inv_len_w, nugget)
                else:
                    length_w = length_full[:Dw]
                    length_z = length_full[Dw:Dw + Dz]
                    inv_len_w = 1.0 / length_w
                    inv_len_z = 1.0 / length_z

                    if parallel:
                        return link_gp_sexp_withz_parallel(m, v, z, overall_input, self.global_input[:, k:], Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget)
                    else:
                        return link_gp_sexp_withz_serial(m, v, z, overall_input, self.global_input[:, k:], Rinv, Rinv_y, scale, inv_len_w, inv_len_z, nugget)
            # ---------- matern2.5 ----------
            else:
                if z is None:
                    length_w = length_full

                    if parallel:
                        return link_gp_matern25_noz_parallel(m, v, overall_input, Rinv, Rinv_y, scale, length_w, nugget)
                    else:
                        return link_gp_matern25_noz_serial(m, v, overall_input, Rinv, Rinv_y, scale, length_w, nugget)
                else:
                    length_w = length_full[:Dw]
                    length_z = length_full[Dw:Dw + Dz]

                    inv_len_z = 1.0 / length_z

                    if parallel:
                        return link_gp_matern25_withz_parallel(m, v, z, overall_input, self.global_input[:, k:], Rinv, Rinv_y, scale, length_w, inv_len_z, nugget)
                    else:
                        return link_gp_matern25_withz_serial(m, v, z, overall_input, self.global_input[:, k:], Rinv, Rinv_y, scale, length_w, inv_len_z, nugget)

    def compute_stats(self):
        """Compute and store key statistics for the GP predictions
        """
        R=self.k_matrix()
        n = self.output.shape[0]
        #U, s, Vh = np.linalg.svd(R)
        #self.Rinv=Vh.T@np.diag(s**-1)@U.T
        #L=np.linalg.cholesky(R)
        #self.Rinv_y=cho_solve((L, True), self.output, check_finite=False)
        #self.Rinv=pinvh(R,check_finite=False)
        #self.Rinv_y=np.dot(self.Rinv,self.output).flatten()
        try:
            L=np.linalg.cholesky(R)
            self.Rinv=cho_solve((L, True), np.eye(n), check_finite=False)
            self.Rinv_y=cho_solve((L, True), self.output[:,0], check_finite=False)
        except LinAlgError:
            self.Rinv=pinvh(R,check_finite=False)
            self.Rinv_y=np.dot(self.Rinv,self.output[:,0])

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