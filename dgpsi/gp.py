import multiprocess.context as ctx
import platform
import numpy as np
from .functions import mice_var
from pathos.multiprocessing import ProcessingPool as Pool
import psutil 
import copy

class gp:
    """
    Class that for Gaussian process emulation.

    Args:
        X (ndarray): a numpy 2d-array where each row is an input data point and 
            each column is an input dimension.
        Y (ndarray): a numpy 2d-array with only one column and each row being an input data point.
        kernel (class): a :class:`.kernel` class that specifies the features of the GP. 
    """

    def __init__(self, X, Y, kernel):
        self.X=X
        self.Y=Y
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        self.kernel=kernel
        self.initialize()
        self.kernel.compute_stats()

    def initialize(self):
        """Assign input/output data to the kernel for training.
        """
        if self.kernel.input_dim is not None:
            self.kernel.input=copy.deepcopy(self.X[:,self.kernel.input_dim])
        else:
            self.kernel.input=copy.deepcopy(self.X)
            self.kernel.input_dim=copy.deepcopy(np.arange(np.shape(self.X)[1]))
        if self.kernel.connect is not None:
            if len(np.intersect1d(self.kernel.connect,self.kernel.input_dim))!=0:
                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
            self.kernel.global_input=copy.deepcopy(self.X[:,self.kernel.connect])
        self.kernel.output=copy.deepcopy(self.Y)
        self.kernel.D=np.shape(self.kernel.input)[1]
        if self.kernel.connect is not None:
            self.kernel.D+=len(self.kernel.connect)
        if self.kernel.prior_name=='ref':
            p=np.shape(self.kernel.input)[1]
            if self.kernel.global_input is not None:
                p+=np.shape(self.kernel.global_input)[1]
            b=1/len(self.kernel.output)**(1/p)*(self.kernel.prior_coef+p)
            self.kernel.prior_coef=np.concatenate((self.kernel.prior_coef, b))
            self.kernel.compute_cl()

    def update_xy(self,X,Y):
        """Update the trained GP emulator with new input and output data without changing the hyperparameter values.

        Args:
            X (ndarray): a numpy 2d-array where each row is an input data point and each column is an input dimension.
            Y (ndarray): a numpy 2d-array with only one column and each row being an input data point.
        """
        self.X=X
        self.Y=Y
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        self.update_kernel()
        self.kernel.compute_stats()
    
    def update_kernel(self):
        """Assign new input/output data to the kernel.
        """
        self.kernel.input=copy.deepcopy(self.X[:,self.kernel.input_dim])
        if self.kernel.connect is not None:
            if len(np.intersect1d(self.kernel.connect,self.kernel.input_dim))!=0:
                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
            self.kernel.global_input=copy.deepcopy(self.X[:,self.kernel.connect])
        self.kernel.output=copy.deepcopy(self.Y)
        if self.kernel.prior_name=='ref':
            self.kernel.compute_cl()

    def train(self):
        """Train the GP model.
        """
        self.kernel.maximise()
        self.kernel.compute_stats()

    def export(self):
        """Export the trained GP.
        """
        final_struct=copy.deepcopy(self.kernel)
        return [final_struct]

    def pmetric(self, x_cand, method='MICE',nugget_s=1.,score_only=False,chunk_num=None,core_num=None):
        """Implement parallel computation of the ALM or MICE criterion for sequential designs.

        Args:
            x_cand, method, nugget_s, score_only: see descriptions of the method :meth:`.gp.metric`.
            chunk_num (int, optional): the number of chunks that the candidate design set **x_cand** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.gp.metric`.
        """
        _, sigma2 = self.ppredict(x=x_cand,chunk_num=chunk_num,core_num=core_num)
        if method == 'ALM':
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,0]
        elif method == 'MICE':
            sigma2_s = mice_var(x_cand, x_cand, copy.deepcopy(self.kernel), nugget_s)
            mice_val = sigma2/sigma2_s
            if score_only:
                return mice_val
            else:
                idx = np.argmax(mice_val, axis=0)
                return idx, mice_val[idx,0]

    def metric(self, x_cand, method='MICE',nugget_s=1.,score_only=False):
        """Compute the value of the ALM or MICE criterion for sequential designs.

        Args:
            x_cand (ndarray): a numpy 2d-array that represents a candidate input design where each row is a design point and 
                each column is a design input dimension.
            method (str, optional): the sequential design approach: MICE (`MICE`) or ALM 
                (`ALM`). Defaults to `MICE`.
            nugget_s (float, optional): the value of the smoothing nugget term used when **method** = '`MICE`'. Defaults to `1.0`.
            score_only (bool, optional): whether to return only the scores of ALM or MICE criterion at all design points contained in **x_cand**.
                Defaults to `False`.

        Returns:
            ndarray_or_tuple: 
            if the argument **score_only** = `True`, a numpy 2d-array is returned that gives the scores of ALM or MICE criterion with rows
                corresponding to design points in the candidate design set **x_cand**

            if the argument **score_only** = `False`, a tuple of two numpy 1d-arrays is returned. The first one gives the index (i.e., row number) 
                of the design point in the candidate design set **x_cand** that has the largest criterion value, 
                which is given by the second element.
        """
        _, sigma2 = self.predict(x=x_cand)
        if method == 'ALM':
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,0]
        elif method == 'MICE':
            sigma2_s = mice_var(x_cand, x_cand, copy.deepcopy(self.kernel), nugget_s)
            mice_val = sigma2/sigma2_s
            if score_only:
                return mice_val
            else:
                idx = np.argmax(mice_val, axis=0)
                return idx, mice_val[idx,0]

    def esloo(self):
        """Compute the (normalised) expected squared LOO of a GP model.

        Returns:
            ndarray: a numpy 2d-array is returned. The array has only one column with its rows corresponding to training data positions.
        """
        mu, sigma2 = self.loo()
        error=(mu-self.Y)**2
        esloo=sigma2+error
        normaliser=2*sigma2**2+4*sigma2*error
        nesloo=esloo/np.sqrt(normaliser)
        return nesloo

    def loo(self, method='mean_var', sample_size=50):
        """Implement the Leave-One-Out cross-validation of a GP model.

        Args:
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach for the LOO. Defaults to `mean_var`.
            sample_size (int, optional): the number of samples to draw from the predictive distribution of
                 GP if **method** = '`sampling`'. Defaults to `50`.

        Returns:
            tuple_or_ndarray: 
            
            if the argument **method** = '`mean_var`', a tuple is returned. The tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has only one column with its rows corresponding to training data positions.

            if the argument **method** = '`sampling`', a numpy 2d-array is returned. The array has its rows corresponding to to training data positions 
                and columns corresponding to `sample_size` number of samples drawn from the predictive distribution of GP.
        """
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

    def ppredict(self,x,method='mean_var',sample_size=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained GP model.

        Args:
            x, method, sample_size: see descriptions of the method :meth:`.gp.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.gp.predict`.
        """
        if platform.system()=='Darwin':
            ctx._force_start_method('forkserver')
        if core_num is None:
            core_num=psutil.cpu_count(logical = False)-1
        if chunk_num is None:
            chunk_num=core_num
        if chunk_num<core_num:
            core_num=chunk_num
        f=lambda x: self.predict(*x) 
        z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            res = pool.map(f, [[x, method, sample_size] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        if method == 'mean_var':
            return tuple(np.concatenate(worker) for worker in zip(*res))
        elif method == 'sampling':
            return np.concatenate(res)

    def predict(self,x,method='mean_var',sample_size=50):
        """Implement predictions from the trained GP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            sample_size (int, optional): the number of samples to draw from the predictive distribution of
                 GP if **method** = '`sampling`'. Defaults to `50`.

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
        if method=='mean_var':
            mu,sigma2=self.kernel.gp_prediction(x=overall_global_test_input[:,self.kernel.input_dim],z=z_k_in)
            return mu.reshape(-1,1), sigma2.reshape(-1,1)
        elif method=='sampling':
            mu,sigma2=self.kernel.gp_prediction(x=overall_global_test_input[:,self.kernel.input_dim],z=z_k_in)
            samples=np.random.normal(mu,np.sqrt(sigma2),size=(sample_size,M)).T
            return samples