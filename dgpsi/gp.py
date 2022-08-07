import multiprocess.context as ctx
import platform
import numpy as np
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
        if core_num==None:
            core_num=psutil.cpu_count(logical = False)-1
        if chunk_num==None:
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
            tuple_or_ndarray: if the argument **method** = '`mean_var`', a tuple is returned:
                the tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has only one column with its rows 
                corresponding to testing positions;

            if the argument **method** = '`sampling`', a numpy 2d-array is returned:
                the array has its rows corresponding to testing positions and columns corresponding to
                sample_size number of samples drawn from the predictive distribution of GP;
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