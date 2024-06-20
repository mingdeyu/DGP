import multiprocess.context as ctx
import platform
import numpy as np
from scipy.spatial.distance import cdist
from .functions import mice_var
from .vecchia import get_pred_nn, loo_gp_vecch
from pathos.multiprocessing import ProcessingPool as Pool
import psutil 
import copy
from numba import set_num_threads

class gp:
    """
    Class that for Gaussian process emulation.

    Args:
        X (ndarray): a numpy 2d-array where each row is an input data point and 
            each column is an input dimension.
        Y (ndarray): a numpy 2d-array with only one column and each row being an input data point.
        kernel (class): a :class:`.kernel` class that specifies the features of the GP. 
        vecchia (bool): a bool indicating if Vecchia approximation will be used. Defaults to `False`. 
        m (int): an integer that gives the size of the conditioning set for the Vecchia approximation in the training. Defaults to `25`. 
        ord_fun (function, optional): a function that decides the ordering of the input of the GP for the Vecchia approximation. If set to `None`, then the default random ordering is used. Defaults to `None`.
    """

    def __init__(self, X, Y, kernel, vecchia=False, m=25, ord_fun=None):
        self.X=X
        self.Y=Y
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        self.kernel=kernel
        self.vecch=vecchia
        self.n_data=self.X.shape[0]
        #if self.n_data>=1e5:
        #    self.kernel.nn_method = 'approx'
        self.m=min(m, self.n_data-1)
        self.ord_fun=ord_fun
        self.initialize()
        if self.vecch:
            self.kernel.ord_nn()
        else:
            self.kernel.compute_stats()
    
    def __setstate__(self, state):
        if 'vecch' not in state:
            state['vecch'] = False
        if 'n_data' not in state:
            state['n_data'] = state['X'].shape[0]
        if 'nn_method' not in state:
            state['nn_method'] = 'exact'
        if 'm' not in state:
            state['m'] = 25
        if 'ord_fun' not in state:
            state['ord_fun'] = None
        self.__dict__.update(state)
        self.kernel.target = 'gp'

    def initialize(self):
        """Assign input/output data to the kernel for training.
        """
        if self.kernel.input_dim is not None:
            self.kernel.input=self.X[:,self.kernel.input_dim]
        else:
            self.kernel.input=(self.X).copy()
            self.kernel.input_dim=np.arange(np.shape(self.X)[1])
        if self.kernel.connect is not None:
            if len(np.intersect1d(self.kernel.connect,self.kernel.input_dim))!=0:
                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
            self.kernel.global_input=self.X[:,self.kernel.connect]
        self.kernel.output=(self.Y).copy()
        self.kernel.D=np.shape(self.kernel.input)[1]
        if self.kernel.connect is not None:
            self.kernel.D+=len(self.kernel.connect)
        #if self.kernel.scale_est:
        #    self.kernel.compute_scale() 
        self.kernel.para_path=np.atleast_2d(np.concatenate((self.kernel.scale,self.kernel.length,self.kernel.nugget)))
        self.kernel.vecch = self.vecch
        self.kernel.m = self.m
        if self.ord_fun is not None:
            self.kernel.ord_fun = self.ord_fun
        if self.kernel.prior_name=='ref':
            p=np.shape(self.kernel.input)[1]
            if self.kernel.global_input is not None:
                p+=np.shape(self.kernel.global_input)[1]
            b=1/self.n_data**(1/p)*(self.kernel.prior_coef+p)
            self.kernel.prior_coef=np.concatenate((self.kernel.prior_coef, b))
            self.kernel.compute_cl()
        self.kernel.target='gp'

    def to_vecchia(self, m=25, ord_fun=None):
        """Convert the GP emulator to the Vecchia mode.

        Args:
            m (int): an integer that gives the size of the conditioning set for the Vecchia approximation in the training. Defaults to `25`. 
            ord_fun (function, optional): a function that decides the ordering of the input of the GP for the Vecchia approximation. If set to `None`, then the default random ordering is used. Defaults to `None`.
        """
        if self.vecch:
            raise Exception('The GP emulator is already in Vecchia mode.')
        else:
            self.vecch=True
            self.m = min(m, self.n_data-1)
            self.ord_fun = ord_fun
            self.kernel.vecch = self.vecch
            self.kernel.m = self.m
            self.kernel.ord_fun = self.ord_fun
            self.kernel.ord_nn()

    def remove_vecchia(self):
        """Remove the Vecchia mode from the GP emulator.
        """
        if self.vecch:
            self.vecch = False
            self.kernel.vecch = self.vecch
            self.kernel.compute_stats()
        else:
            raise Exception('The GP emulator is already in non-Vecchia mode.')

    def update_xy(self, X, Y, reset=False):
        """Update the trained GP emulator with new input and output data.

        Args:
            X (ndarray): a numpy 2d-array where each row is an input data point and each column is an input dimension.
            Y (ndarray): a numpy 2d-array with only one column and each row being an input data point.
            reset (bool, optional): whether to reset hyperparameter values of the GP emulator. Defaults to `False`. 
        """
        self.X=X
        self.Y=Y
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        self.n_data=self.X.shape[0]
        #if self.n_data>=1e5:
        #    self.kernel.nn_method = 'approx'
        self.m=min(self.m, self.n_data-1)
        self.update_kernel(reset_lengthscale=reset)
        if self.vecch:
            self.kernel.ord_nn()
        else:
            self.kernel.compute_stats()
    
    def update_kernel(self, reset_lengthscale):
        """Assign new input/output data to the kernel.
        Args: 
            reset_lengthscale (bool): whether to reset hyperparameter of the GP emulator to the initial values.
        """
        self.kernel.input=self.X[:,self.kernel.input_dim]
        if self.kernel.connect is not None:
            if len(np.intersect1d(self.kernel.connect,self.kernel.input_dim))!=0:
                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
            self.kernel.global_input=self.X[:,self.kernel.connect]
        self.kernel.output=self.Y.copy()
        self.kernel.m = self.m
        if reset_lengthscale:
            initial_hypers=self.kernel.para_path[0,:]
            self.kernel.scale=initial_hypers[[0]]
            self.kernel.length=initial_hypers[1:-1]
            self.kernel.nugget=initial_hypers[[-1]]
        if self.kernel.prior_name=='ref':
            self.kernel.compute_cl()

    def train(self):
        """Train the GP model.
        """
        self.kernel.maximise()
        if not self.vecch:
            self.kernel.compute_stats()

    def export(self):
        """Export the trained GP.
        """
        final_struct=copy.deepcopy(self.kernel)
        return [final_struct]

    def pmetric(self, x_cand, method='MICE',nugget_s=1.,m=50,score_only=False,chunk_num=None,core_num=None):
        """Implement parallel computation of the ALM, MICE, or VIGF criterion for sequential designs.

        Args:
            x_cand, method, nugget_s, m, score_only: see descriptions of the method :meth:`.gp.metric`.
            chunk_num (int, optional): the number of chunks that the candidate design set **x_cand** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.gp.metric`.
        """
        if method == 'ALM':
            _, sigma2 = self.ppredict(x=x_cand,m=m,chunk_num=chunk_num,core_num=core_num)
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,0]
        elif method == 'MICE':
            _, sigma2 = self.ppredict(x=x_cand,m=m,chunk_num=chunk_num,core_num=core_num)
            sigma2_s = mice_var(x_cand, x_cand, self.kernel.input_dim, self.kernel.connect, self.kernel.name, self.kernel.length, self.kernel.scale, self.kernel.nugget[0], nugget_s)
            mice_val = sigma2/sigma2_s
            if score_only:
                return mice_val
            else:
                idx = np.argmax(mice_val, axis=0)
                return idx, mice_val[idx,0]
        elif method == 'VIGF':
            X0 = np.unique(self.X, axis=0)
            if len(X0) != self.n_data:
                raise Exception('VIGF criterion is currently not applicable to GP emulators whose training data contain replicates.')
            if self.vecch or self.n_data>500:
                index = get_pred_nn(x_cand, self.X, 1, method = self.kernel.nn_method).flatten()
            else:
                Dist=cdist(x_cand, self.X, "euclidean")
                index=np.argmin(Dist, axis=1)
            mu, sigma2 = self.ppredict(x=x_cand,m=m,chunk_num=chunk_num,core_num=core_num)
            bias=(mu-self.Y[index,:])**2
            vigf=4*sigma2*bias+2*sigma2**2
            if score_only:
                return vigf
            else:
                idx = np.argmax(vigf, axis=0)
                return idx, vigf[idx,0]
            
    def metric(self, x_cand, method='MICE',nugget_s=1.,m=50,score_only=False):
        """Compute the value of the ALM, MICE, or VIGF criterion for sequential designs.

        Args:
            x_cand (ndarray): a numpy 2d-array that represents a candidate input design where each row is a design point and 
                each column is a design input dimension.
            method (str, optional): the sequential design approach: MICE (`MICE`), ALM 
                (`ALM`) or VIGF (`VIGF`). Defaults to `MICE`.
            nugget_s (float, optional): the value of the smoothing nugget term used when **method** = '`MICE`'. Defaults to `1.0`.
            m (int, optional): the size of the conditioning set for metric calculations if the GP was built under the Vecchia approximation. Defaults to `50`.
            score_only (bool, optional): whether to return only the scores of ALM or MICE criterion at all design points contained in **x_cand**.
                Defaults to `False`.

        Returns:
            ndarray_or_tuple: 
            if the argument **score_only** = `True`, a numpy 2d-array is returned that gives the scores of ALM, MICE, or VIGF criterion with rows
                corresponding to design points in the candidate design set **x_cand**

            if the argument **score_only** = `False`, a tuple of two numpy 1d-arrays is returned. The first one gives the index (i.e., row number) 
                of the design point in the candidate design set **x_cand** that has the largest criterion value, 
                which is given by the second element.
        """
        if method == 'ALM':
            _, sigma2 = self.predict(x=x_cand, m=m)
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,0]
        elif method == 'MICE':
            _, sigma2 = self.predict(x=x_cand, m=m)
            sigma2_s = mice_var(x_cand, x_cand, self.kernel.input_dim, self.kernel.connect, self.kernel.name, self.kernel.length, self.kernel.scale, self.kernel.nugget[0], nugget_s)
            mice_val = sigma2/sigma2_s
            if score_only:
                return mice_val
            else:
                idx = np.argmax(mice_val, axis=0)
                return idx, mice_val[idx,0]
        elif method == 'VIGF':
            X0 = np.unique(self.X, axis=0)
            if len(X0) != self.n_data:
                raise Exception('VIGF criterion is currently not applicable to GP emulators whose training data contain replicates.')
            if self.vecch or self.n_data>500:
                index = get_pred_nn(x_cand, self.X, 1, method = self.kernel.nn_method).flatten()
            else:
                Dist=cdist(x_cand, self.X, "euclidean")
                index=np.argmin(Dist, axis=1)
            mu, sigma2 = self.predict(x=x_cand, m=m)
            bias=(mu-self.Y[index,:])**2
            vigf=4*sigma2*bias+2*sigma2**2
            if score_only:
                return vigf
            else:
                idx = np.argmax(vigf, axis=0)
                return idx, vigf[idx,0]

    def esloo(self, m=30):
        """Compute the (normalised) expected squared LOO of a GP model.

        Args:
            m (int, optional): the size of the conditioning set for loo calculations involved under the Vecchia approximation. Defaults to `30`.

        Returns:
            ndarray: a numpy 2d-array is returned. The array has only one column with its rows corresponding to training data positions.
        """
        mu, sigma2 = self.loo(m=m)
        error=(mu-self.Y)**2
        esloo=sigma2+error
        normaliser=2*sigma2**2+4*sigma2*error
        nesloo=esloo/np.sqrt(normaliser)
        return nesloo

    def loo(self, method='mean_var', sample_size=50, m=30):
        """Implement the Leave-One-Out cross-validation of a GP model.

        Args:
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach for the LOO. Defaults to `mean_var`.
            sample_size (int, optional): the number of samples to draw from the predictive distribution of
                 GP if **method** = '`sampling`'. Defaults to `50`.
            m (int, optional): the size of the conditioning set for loo calculations if the GP was built under the Vecchia approximation. Defaults to `30`.

        Returns:
            tuple_or_ndarray: 
            
            if the argument **method** = '`mean_var`', a tuple is returned. The tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has only one column with its rows corresponding to training data positions.

            if the argument **method** = '`sampling`', a numpy 2d-array is returned. The array has its rows corresponding to to training data positions 
                and columns corresponding to `sample_size` number of samples drawn from the predictive distribution of GP.
        """
        if self.vecch:
            X_scale = self.X/self.kernel.length
            NNarray = get_pred_nn(X_scale, X_scale, m+1, method=self.kernel.nn_method)
            mu,sigma2 = loo_gp_vecch(self.X, NNarray, self.Y, self.kernel.scale[0], self.kernel.length, self.kernel.nugget[0], self.kernel.name)
            mu,sigma2 = mu.reshape(-1,1), sigma2.reshape(-1,1)
        else:
            scale = self.kernel.scale
            Rinv = self.kernel.Rinv
            Rinv_y = self.kernel.Rinv_y[:,np.newaxis]
            sigma2 = (1/np.diag(Rinv)).reshape(-1,1)
            mu = self.Y - Rinv_y*sigma2
            sigma2 = scale*sigma2
        if method=='mean_var':
            return mu, sigma2
        elif method=='sampling':
            samples=np.random.normal(mu.flatten(),np.sqrt(sigma2.flatten()),size=(sample_size,len(mu))).T
            return samples

    def ppredict(self,x,method='mean_var',sample_size=50,m=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained GP model.

        Args:
            x, method, sample_size, m: see descriptions of the method :meth:`.gp.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.gp.predict`.
        """
        os_type = platform.system()
        if os_type in ['Darwin', 'Linux']:
            ctx._force_start_method('forkserver')
        total_cores = psutil.cpu_count(logical = False)
        if core_num is None:
            core_num = total_cores//2
        if chunk_num is None:
            chunk_num=core_num
        if chunk_num<core_num:
            core_num=chunk_num
        num_thread = total_cores // core_num
        def f(params):
            x, method, sample_size, m = params
            set_num_threads(num_thread)
            return self.predict(x, method, sample_size, m)
        z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            res = pool.map(f, [[x, method, sample_size, m] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        if method == 'mean_var':
            return tuple(np.concatenate(worker) for worker in zip(*res))
        elif method == 'sampling':
            return np.concatenate(res)

    def predict(self,x,method='mean_var',sample_size=50,m=50):
        """Implement predictions from the trained GP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            sample_size (int, optional): the number of samples to draw from the predictive distribution of
                 GP if **method** = '`sampling`'. Defaults to `50`.
            m (int, optional): the size of the conditioning set for predictions if the GP was built under the Vecchia approximation. Defaults to `50`.

        Returns:
            tuple_or_ndarray: 
            
            if the argument **method** = '`mean_var`', a tuple is returned:

                the tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has only one column with its rows 
                corresponding to testing positions.

            if the argument **method** = '`sampling`', a numpy 2d-array is returned:

                the array has its rows corresponding to testing positions and columns corresponding to
                `sample_size` number of samples drawn from the predictive distribution of GP.
        """
        if x.ndim==1:
            raise Exception('The testing input has to be a numpy 2d-array')
        M=len(x)
        overall_global_test_input=x
        if self.kernel.connect is not None:
            z_k_in=overall_global_test_input[:,self.kernel.connect]
        else:
            z_k_in=None
        self.kernel.pred_m = m
        if method=='mean_var':
            mu,sigma2=self.kernel.gp_prediction(x=overall_global_test_input[:,self.kernel.input_dim],z=z_k_in)
            return mu.reshape(-1,1), sigma2.reshape(-1,1)
        elif method=='sampling':
            mu,sigma2=self.kernel.gp_prediction(x=overall_global_test_input[:,self.kernel.input_dim],z=z_k_in)
            samples=np.random.normal(mu,np.sqrt(sigma2),size=(sample_size,M)).T
            return samples