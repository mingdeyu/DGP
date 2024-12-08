import multiprocess.context as ctx
import platform
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil   
from .imputation import imputer
import copy
from scipy.spatial.distance import cdist
from .functions import ghdiag, mice_var, esloo_calculation, logloss
from .vecchia import get_pred_nn
from contextlib import contextmanager
from numba import set_num_threads

class emulator:
    """Class to make predictions from the trained DGP model.

    Args:
        all_layer (list): a list that contains the trained DGP model produced by the method :meth:`.estimate`
            of the :class:`.dgp` class. 
        N (int, optional): the number of imputations to produce the predictions. Increase the value to account for
            more imputation uncertainties. Defaults to `10`.
        block (bool, optional): whether to use the blocked (layer-wise) ESS for the imputations. Defaults to `True`.
    """
    def __init__(self, all_layer, N=10, block=True):
        self.all_layer=all_layer
        self.n_layer=len(all_layer)
        if self.all_layer[0][0].vecch:
            self.vecch=True
        else:
            self.vecch=False
        self.imp=imputer(self.all_layer, block)
        if self.vecch:
            (self.imp).update_ord_nn()
            (self.imp).sample(burnin=20)
        else:
            (self.imp).sample(burnin=50)
        self.all_layer_set=[]
        for _ in range(N):
            if self.vecch:
                (self.imp).update_ord_nn()
            (self.imp).sample()
            if not self.vecch:
                (self.imp).key_stats()
            (self.all_layer_set).append(copy.deepcopy(self.all_layer))
        #self.nb_parallel=nb_parallel
        #if len(self.all_layer[0][0].input)>=500 and self.nb_parallel==False:
        #    print('Your training data size is greater than %i, you might want to set "nb_parallel=True" to accelerate the prediction.' % (500))
    
    #def set_nb_parallel(self,nb_parallel):
    #    """Set **self.nb_parallel** to the bool value given by **nb_parallel**. This method is useful to change **self.nb_parallel**
    #        when the :class:`.emulator` class has already been built.
    #    """
    #    self.nb_parallel=nb_parallel
    def __setstate__(self, state):
        if 'all_layer_set_copy' in state:
            del state['all_layer_set_copy']
        if 'vecch' not in state:
            state['vecch'] = False
        if 'nb_parallel' in state:
            del state['nb_parallel']
        self.__dict__.update(state)
            
    def to_vecchia(self):
        """Convert the DGP emulator to the Vecchia mode.
        """
        if self.vecch:
            raise Exception('The DGP emulator is already in Vecchia mode.')
        else:
            self.vecch=True
            for one_imputed_layer in self.all_layer_set:
                for layer in one_imputed_layer:
                    for kernel in layer:
                        if kernel.type == 'gp':
                            kernel.vecch = self.vecch

    def remove_vecchia(self):
        """Remove the Vecchia mode from the DGP emulator.
        """
        if self.vecch:
            self.vecch = False
            for one_imputed_layer in self.all_layer_set:
                for layer in one_imputed_layer:
                    for kernel in layer:
                        if kernel.type == 'gp':
                            kernel.vecch = self.vecch
                            kernel.compute_stats()
        else:
            raise Exception('The DGP emulator is already in non-Vecchia mode.')
        
    def esloo(self, X, Y, m=30):
        """Compute the (normalised) expected squared LOO from a DGP emulator.

        Args:
            X (ndarray): the training input data used to build the DGP emulator via the :class:`.dgp` class.
            Y (ndarray): the training output data used to build the DGP emulator via the :class:`.dgp` class.
            m (int, optional): the size of the conditioning set for loo calculations if the GP was built under the Vecchia approximation. Defaults to `30`.

        Returns:
            ndarray: a numpy 2d-array is returned. The array has its rows corresponding to training input
                positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer);
        """
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
            _, counts = np.unique(indices, return_counts=True)
            start_rows = np.cumsum(np.concatenate(([0], counts[:-1])))
        else:
            indices = None
            start_rows = np.arange(len(X))
        m_pred = m+1 if self.vecch else X.shape[0]
        with self.change_vecch_state():
            mu_i, var_i = self.predict(X, aggregation=False, m=m_pred)
        mu_i, var_i = np.stack(mu_i), np.stack(var_i)
        final_res = esloo_calculation(mu_i, var_i, Y, indices, start_rows)
        return final_res

    def pesloo(self, X, Y, m=30, core_num=None):
        """Compute in parallel the (normalised) expected squared LOO from a DGP emulator.

        Args:
            X, Y, m: see descriptions of the method :meth:`.emulator.esloo`.
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.emulator.esloo`.
        """
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
            _, counts = np.unique(indices, return_counts=True)
            start_rows = np.cumsum(np.concatenate(([0], counts[:-1])))
        else:
            indices = None
            start_rows = np.arange(len(X))
        m_pred = m+1 if self.vecch else X.shape[0]
        with self.change_vecch_state():
            mu_i, var_i = self.ppredict(X, aggregation=False, m=m_pred, core_num=core_num)
        mu_i, var_i = np.stack(mu_i), np.stack(var_i)
        final_res = esloo_calculation(mu_i, var_i, Y, indices, start_rows)
        return final_res
        
    @contextmanager
    def change_vecch_state(self):
        for one_imputed_layer in self.all_layer_set:
            for layer in one_imputed_layer:
                for kernel in layer:
                    if kernel.type == 'gp':
                        if not self.vecch:
                            kernel.vecch = True
                        kernel.loo_state = True
        yield
        # Restore original state
        for one_imputed_layer in self.all_layer_set:
            for layer in one_imputed_layer:
                for kernel in layer:
                    if kernel.type == 'gp':
                        if not self.vecch:
                            kernel.vecch = False
                        kernel.loo_state = False
        
    def loo(self, X, method=None, mode = 'prob', sample_size=50, m=30):
        """Implement the Leave-One-Out cross-validation from a DGP emulator.

        Args:
            X (ndarray): the training input data used to build the DGP emulator via the :class:`.dgp` class.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach for the LOO. If set to None, sampling 
                (`sampling`) approach is used for DGP emulators with a categorical likelihood. Otherwise, 
                mean-variance (`mean_var`) approach is used. mean-variance (`mean_var`) approach is not applicable
                to DGP emulators with a categorical likelihood. Defaults to None.
            mode (str, optional): whether to return samples of probabilities of classes (`prob`) or the classes themselves (`label`). Defaults to `prob`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                 Defaults to `50`.
            m (int, optional): the size of the conditioning set for loo calculations if the GP was built under the Vecchia approximation. Defaults to `30`.
            
        Returns:
            tuple_or_list: 
            if the argument **method** = '`mean_var`', a tuple is returned. The tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has its rows corresponding to training input
                positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer);
            
            If the argument **method** = '`sampling`', the function returns a list. This list contains *D* elements, where *D* represents either the number 
            of GP/likelihood nodes in the final layer or the number of classes (when **mode** = '`prob`' and the emulator uses a categorical likelihood). 
            Each element in the list is a 2d-array in which rows correspond to training input positions, and columns represent samples of size **N** * **sample_size**.
        """
        if method is None:
            if self.all_layer[-1][0].name == 'Categorical':
                method = 'sampling'
            else:
                method = 'mean_var'
        else:
            if self.all_layer[-1][0].name == 'Categorical' and method == 'mean_var':
                raise Exception("The method argument must be 'sampling' when the DGP emulator has a categorical likelihood layer.")
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        m_pred = m+1 if self.vecch else X.shape[0]
        with self.change_vecch_state():
            if self.all_layer[-1][0].name == 'Categorical':
                final_res = self.classify(X, mode = mode, sample_size=sample_size, m=m_pred)
            else:
                final_res = self.predict(X, method=method, sample_size=sample_size, m=m_pred)
        if isrep:
            modified_items = [item[indices, :] for item in final_res]
            final_res = type(final_res)(modified_items)
        return final_res
    
    def ploo(self, X, method=None, mode = 'prob', sample_size=50, m=30, core_num=None):
        """Implement the parallel Leave-One-Out cross-validation from a DGP emulator.

        Args:
            X, method, mode, sample_size, m: see descriptions of the method :meth:`.emulator.loo`.
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.emulator.loo`.
        """
        if method is None:
            if self.all_layer[-1][0].name == 'Categorical':
                method = 'sampling'
            else:
                method = 'mean_var'
        else:
            if self.all_layer[-1][0].name == 'Categorical' and method == 'mean_var':
                raise Exception("The method argument must be 'sampling' when the DGP emulator has a categorical likelihood layer.")
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        m_pred = m+1 if self.vecch else X.shape[0]
        with self.change_vecch_state():
            if self.all_layer[-1][0].name == 'Categorical':
                final_res = self.pclassify(X, mode = mode, sample_size=sample_size, m=m_pred, core_num=core_num)
            else:
                final_res = self.ppredict(X, method=method, sample_size=sample_size, m=m_pred, core_num=core_num)
        if isrep:
            modified_items = [item[indices, :] for item in final_res]
            final_res = type(final_res)(modified_items)
        return final_res

    def pmetric(self, x_cand, method='ALM', obj=None, nugget_s=1.,m=50,score_only=False,chunk_num=None,core_num=None):
        """Compute the value of the ALM or MICE criterion for sequential designs in parallel.

        Args:
            x_cand, method, obj, nugget_s, m, score_only: see descriptions of the method :meth:`.emulator.metric`.
            chunk_num (int, optional): the number of chunks that the candidate design set **x_cand** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.emulator.metric`.
        """
        if x_cand.ndim==1:
            raise Exception('The candidate design set has to be a numpy 2d-array.')
        islikelihood = True if self.all_layer[self.n_layer-1][0].type=='likelihood' else False
        #if self.all_layer[self.n_layer-1][0].type=='likelihood':
        #    raise Exception('The method is only applicable to DGPs without likelihood layers.')
        if method == 'ALM':
            if islikelihood:
                if self.all_layer[-1][0].name=='Categorical':
                    _, sigma2, _ = self.pclassify(x=x_cand,method='mean_var',full_layer=True, m=m, chunk_num=chunk_num,core_num=core_num)
                    sigma2 = sigma2[-1]
                else:
                    _, sigma2 = self.ppredict(x=x_cand,full_layer=True,m=m,chunk_num=chunk_num,core_num=core_num)
                    sigma2 = sigma2[-2]
            else:
                _, sigma2 = self.ppredict(x=x_cand,chunk_num=chunk_num,core_num=core_num)
            if score_only:
                return sigma2 
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,np.arange(sigma2.shape[1])]
        elif method == 'MICE':
            os_type = platform.system()
            if os_type in ['Darwin', 'Linux']:
                ctx._force_start_method('forkserver')
            total_cores = psutil.cpu_count(logical = False)
            if core_num is None:
                core_num=total_cores//2
            if chunk_num is None:
                chunk_num=core_num
            if chunk_num<core_num:
                core_num=chunk_num
            num_thread = total_cores // core_num
            if islikelihood and self.n_layer==2:
                def f(params):
                    x_cand,m = params
                    set_num_threads(num_thread)
                    return self.predict_mice_2layer_likelihood(x_cand,m)
                z=np.array_split(x_cand,chunk_num)
                with Pool(core_num) as pool:
                    res = pool.map(f, [[x, m] for x in z])
                    pool.close()
                    pool.join()
                    pool.clear()
                sigma2 = np.concatenate(res)
                M=len(x_cand)
                last_layer = self.all_layer[0]
                D=len(last_layer)
                sigma2_s=np.empty((M,D))
                for k in range(D):
                    kernel = last_layer[k]
                    sigma2_s[:,k] = mice_var(x_cand, x_cand, kernel.input_dim, kernel.connect, kernel.name, kernel.length, kernel.scale, kernel.nugget[0], nugget_s).flatten()
                avg_mice = sigma2/sigma2_s
            else:
                def f(params):
                    x, islikelihood, m = params
                    set_num_threads(num_thread)
                    return self.predict_mice(x, islikelihood, m)
                z=np.array_split(x_cand,chunk_num)
                with Pool(core_num) as pool:
                    res = pool.map(f, [[x, islikelihood, m] for x in z])
                    pool.close()
                    pool.join()
                    pool.clear()
                combined_res=[]
                for element in zip(*res):
                    combined_res.append(list(np.concatenate(workers) for workers in zip(*list(element))))
                predicted_input, sigma2 = combined_res[0], combined_res[1]   
                M=len(x_cand)
                D=len(self.all_layer[-2]) if islikelihood else len(self.all_layer[-1])
                mice=np.zeros((M,D))
                S=len(self.all_layer_set)
                for i in range(S):
                    last_layer=self.all_layer_set[i][-2] if islikelihood else self.all_layer_set[i][-1]
                    sigma2_s_i=np.empty((M,D))
                    for k in range(D):
                        kernel = last_layer[k]
                        sigma2_s_i[:,k] = mice_var(predicted_input[i], x_cand, kernel.input_dim, kernel.connect, kernel.name, kernel.length, kernel.scale, kernel.nugget[0], nugget_s).flatten()
                    with np.errstate(divide='ignore'):
                        mice += np.log(sigma2[i]/sigma2_s_i)
                avg_mice=mice/S
            if score_only:
                return avg_mice
            else:
                idx = np.argmax(avg_mice, axis=0)
                return idx, avg_mice[idx,np.arange(avg_mice.shape[1])]
        elif method == 'VIGF':
            os_type = platform.system()
            if os_type in ['Darwin', 'Linux']:
                ctx._force_start_method('forkserver')
            total_cores = psutil.cpu_count(logical = False)
            if core_num is None:
                core_num=total_cores//2
            if chunk_num is None:
                chunk_num=core_num
            if chunk_num<core_num:
                core_num=chunk_num
            num_thread = total_cores // core_num
            if obj is None:
                raise Exception('The dgp object that is used to build the emulator must be supplied to the argument `obj` when VIGF criterion is chosen.')
            if islikelihood is not True and obj.indices is not None:
                raise Exception('VIGF criterion is currently not applicable to DGP emulators whose training data contain replicates but without a likelihood node.')
            X=obj.X
            if obj.vecch or obj.n_data>500:
                index = get_pred_nn(x_cand, X, 1, method = obj.nn_method).flatten()
            else:
                Dist=cdist(x_cand, X, "euclidean")
                index=np.argmin(Dist, axis=1)
            if islikelihood and self.n_layer==2:
                def f(params):
                    x, index, m = params
                    set_num_threads(num_thread)
                    return self.predict_vigf_2layer_likelihood(x, index, m)
                z=np.array_split(x_cand,chunk_num)
                sub_indx=np.array_split(index,chunk_num)
                with Pool(core_num) as pool:
                    res = pool.map(f, [[x, index, m] for x,index in zip(z,sub_indx)])
                    pool.close()
                    pool.join()
                    pool.clear()
            else:
                def f(params):
                    x, index, islikelihood, m = params
                    set_num_threads(num_thread)
                    return self.predict_vigf(x, index, islikelihood, m)
                z=np.array_split(x_cand,chunk_num)
                sub_indx=np.array_split(index,chunk_num)
                with Pool(core_num) as pool:
                    res = pool.map(f, [[x, index, islikelihood, m] for x,index in zip(z,sub_indx)])
                    pool.close()
                    pool.join()
                    pool.clear()
            combined_res=[]
            for element in zip(*res):
                combined_res.append(list(np.concatenate(workers) for workers in zip(*list(element))))
            bias, sigma2 = np.asarray(combined_res[0]), np.asarray(combined_res[1])
            E1=np.mean(np.square(bias)+6*bias*sigma2+3*np.square(sigma2),axis=0)
            E2=np.mean(bias+sigma2, axis=0)
            vigf=E1-E2**2  
            if score_only:
                return vigf
            else:
                idx = np.argmax(vigf, axis=0)
                return idx, vigf[idx,np.arange(vigf.shape[1])]

    def metric(self, x_cand, method='ALM', obj=None, nugget_s=1.,m=50,score_only=False):
        """Compute the value of the ALM, MICE, or VIGF criterion for sequential designs.

        Args:
            x_cand (ndarray): a numpy 2d-array that represents a candidate input design where each row is a design point and 
                each column is a design input dimension.
            method (str, optional): the sequential design approach: MICE (`MICE`), ALM 
                (`ALM`), or VIGF (`VIGF`). Defaults to `ALM`.
            obj (class, optional): the dgp object that is used to build the DGP emulator when **method** = '`VIGF`'. Defaults to `None`.
            nugget_s (float, optional): the value of the smoothing nugget term used when **method** = '`MICE`'. Defaults to `1.0`.
            m (int, optional): the size of the conditioning set for metric calculations if the DGP was built under the Vecchia approximation. Defaults to `50`.
            score_only (bool, optional): whether to return only the scores of ALM or MICE criterion at all design points contained in **x_cand**.
                Defaults to `False`.

        Returns:
            ndarray_or_tuple: 
            if the argument **score_only** = `True`, a numpy 2d-array is returned that gives the scores of ALM, MICE, or VIGF criterion with rows
               corresponding to design points in the candidate design set **x_cand** and columns corresponding to output dimensions;

            if the argument **score_only** = `False`, a tuple of two numpy 1d-arrays is returned. The first one gives the indices (i.e., row numbers) 
                of the design points in the candidate design set **x_cand** that have the largest criterion values, which are given by the second array, 
                across different outputs of the DGP emulator.
        """
        if x_cand.ndim==1:
            raise Exception('The candidate design set has to be a numpy 2d-array.')
        islikelihood = True if self.all_layer[self.n_layer-1][0].type=='likelihood' else False
        #    raise Exception('The method is only applicable to DGPs without likelihood layers.')
        if method == 'ALM':
            if islikelihood:
                if self.all_layer[-1][0].name=='Categorical':
                    _, sigma2, _ = self.classify(x=x_cand,method = 'mean_var', full_layer=True, m=m)
                    sigma2 = sigma2[-1]
                else:
                    _, sigma2 = self.predict(x=x_cand,full_layer=True, m=m)
                    sigma2 = sigma2[-2]
            else:
                _, sigma2 = self.predict(x=x_cand, m=m)
            #if self.all_layer[self.n_layer-1][0].type=='likelihood':
            #    _, sigma2 = self.predict(x=x_cand,full_layer=True)
            #else:
            #    _, sigma2 = self.predict(x=x_cand)
            if score_only:
                return sigma2 
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,np.arange(sigma2.shape[1])]
        elif method == 'MICE':
            if islikelihood and self.n_layer==2:
                sigma2 = self.predict_mice_2layer_likelihood(x_cand, m=m)
                M=len(x_cand)
                last_layer = self.all_layer[0]
                D=len(last_layer)
                sigma2_s=np.empty((M,D))
                for k in range(D):
                    kernel = last_layer[k]
                    sigma2_s[:,k] = mice_var(x_cand, x_cand, kernel.input_dim, kernel.connect, kernel.name, kernel.length, kernel.scale, kernel.nugget[0], nugget_s).flatten()
                avg_mice = sigma2/sigma2_s
            else:
                predicted_input, sigma2 = self.predict_mice(x_cand, islikelihood, m=m)
                M=len(x_cand)
                D=len(self.all_layer[-2]) if islikelihood else len(self.all_layer[-1])
                mice=np.zeros((M,D))
                S=len(self.all_layer_set)
                for i in range(S):
                    last_layer=self.all_layer_set[i][-2] if islikelihood else self.all_layer_set[i][-1]
                    sigma2_s_i=np.empty((M,D))
                    for k in range(D):
                        kernel = last_layer[k]
                        sigma2_s_i[:,k] = mice_var(predicted_input[i], x_cand, kernel.input_dim, kernel.connect, kernel.name, kernel.length, kernel.scale, kernel.nugget[0], nugget_s).flatten()
                    with np.errstate(divide='ignore'):
                        mice += np.log(sigma2[i]/sigma2_s_i)
                avg_mice=mice/S
            if score_only:
                return avg_mice
            else:
                idx = np.argmax(avg_mice, axis=0)
                return idx, avg_mice[idx,np.arange(avg_mice.shape[1])]
        elif method == 'VIGF':
            #To-do for the follow case
            if obj is None:
                raise Exception('The dgp object that is used to build the emulator must be supplied to the argument `obj` when VIGF criterion is chosen.')
            if islikelihood is not True and obj.indices is not None:
                raise Exception('VIGF criterion is currently not applicable to DGP emulators whose training data contain replicates but without a likelihood node.')
            X=obj.X
            if obj.vecch or obj.n_data>500:
                index = get_pred_nn(x_cand, X, 1, method = obj.nn_method).flatten()
            else:
                Dist=cdist(x_cand, X, "euclidean")
                index=np.argmin(Dist, axis=1)
            if islikelihood and self.n_layer==2:
                bias, sigma2 = self.predict_vigf_2layer_likelihood(x_cand, index, m=m)
            else:
                bias, sigma2 = self.predict_vigf(x_cand, index, islikelihood, m=m)
            bias, sigma2 = np.asarray(bias), np.asarray(sigma2)    
            E1=np.mean(np.square(bias)+6*bias*sigma2+3*np.square(sigma2),axis=0)
            E2=np.mean(bias+sigma2, axis=0)
            vigf=E1-E2**2
            if score_only:
                return vigf
            else:
                idx = np.argmax(vigf, axis=0)
                return idx, vigf[idx,np.arange(vigf.shape[1])]

    def predict_mice_2layer_likelihood(self,x_cand,m):
        """Implement predictions from the trained DGP model with 2 layers (including a likelihood layer) that are required to calculate the MICE criterion.
        """
        M=len(x_cand)
        layer=self.all_layer[0]
        D=len(layer)
        #start calculation
        variance_pred=np.empty((M,D))
        for k in range(D):
            kernel=layer[k]
            kernel.pred_m = m
            if kernel.connect is not None:
                z_k_in=x_cand[:,kernel.connect]
            else:
                z_k_in=None
            _,v_k=kernel.gp_prediction(x=x_cand[:,kernel.input_dim],z=z_k_in)
            variance_pred[:,k]=v_k
        return variance_pred
            
    def predict_mice(self,x_cand,islikelihood,m):
        """Implement predictions from the trained DGP model that are required to calculate the MICE criterion.
        """
        S=len(self.all_layer_set)
        M=len(x_cand)
        D=len(self.all_layer[-2]) if islikelihood else len(self.all_layer[-1])
        N_layer=self.n_layer-1 if islikelihood else self.n_layer
        variance_pred_set=[]
        pred_input_set=[]
        #start calculation
        for i in range(S):
            one_imputed_all_layer=self.all_layer_set[i]
            variance_pred=np.empty((M,D))
            overall_global_test_input=x_cand
            for l in range(N_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                elif l==N_layer-1:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        _,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        variance_pred[:,k]=v_k
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            variance_pred_set.append(variance_pred)
            pred_input_set.append(overall_test_input_mean)
        return pred_input_set, variance_pred_set

    def predict_vigf_2layer_likelihood(self,x_cand,index,m):
        """Implement predictions from the trained DGP model with 2 layers (including a likelihood layer) that are required to calculate the VIGF criterion.
        """
        S=len(self.all_layer_set)
        M=len(x_cand)
        #start calculation
        bias_pred_set=[]
        variance_pred_set=[]
        for i in range(S):
            one_imputed_all_layer=self.all_layer_set[i]
            layer=one_imputed_all_layer[0]
            D=len(layer)
            bias_pred=np.empty((M,D))
            variance_pred=np.empty((M,D))
            for k in range(D):
                kernel=layer[k]
                kernel.pred_m = m
                if kernel.connect is not None:
                    z_k_in=x_cand[:,kernel.connect]
                else:
                    z_k_in=None
                m_k,v_k=kernel.gp_prediction(x=x_cand[:,kernel.input_dim],z=z_k_in)
                bias_pred[:,k]=(m_k-kernel.output[index,:].flatten())**2
                variance_pred[:,k]=v_k
            bias_pred_set.append(bias_pred)
            variance_pred_set.append(variance_pred)
        return bias_pred_set, variance_pred_set

    def predict_vigf(self,x_cand,index,islikelihood,m):
        """Implement predictions from the trained DGP model that are required to calculate the VIGF criterion.
        """
        S=len(self.all_layer_set)
        M=len(x_cand)
        N_layer=self.n_layer-1 if islikelihood else self.n_layer
        bias_pred_set=[]
        variance_pred_set=[]
        #input_mean_pred_set=[]
        #input_variance_pred_set=[]
        #start calculation
        for i in range(S):
            one_imputed_all_layer=self.all_layer_set[i]
            overall_global_test_input=x_cand
            for l in range(N_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        if l!=N_layer-1:
                            overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                        else:
                            overall_test_output_mean[:,k],overall_test_output_var[:,k]=(m_k-kernel.output[index,:].flatten())**2,v_k
                    if l!=N_layer-1:
                        overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            bias_pred_set.append(overall_test_output_mean)
            variance_pred_set.append(overall_test_output_var)
            #input_mean_pred_set.append(overall_test_input_mean)
            #input_variance_pred_set.append(overall_test_input_var)
        return bias_pred_set,variance_pred_set

    def ppredict(self,x,method='mean_var',full_layer=False,sample_size=50,m=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained DGP model.

        Args:
            x, method, full_layer, sample_size, m: see descriptions of the method :meth:`.emulator.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.emulator.predict`.
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
            x_chunk, method, full_layer, sample_size, m, aggregation = params
            set_num_threads(num_thread)
            return self.predict(x_chunk, method, full_layer, sample_size, m, aggregation)
        z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            #pool.restart()
            res = pool.map(f, [[x, method, full_layer, sample_size, m, True] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        if method == 'mean_var':
            if full_layer:
                combined_res=[]
                for layer in zip(*res):
                    combined_res.append(list(np.concatenate(workers) for workers in zip(*list(layer))))
                return tuple(combined_res)
            else:
                return tuple(np.concatenate(worker) for worker in zip(*res))
        elif method == 'sampling':
            if full_layer:
                combined_res=[]
                for layer in zip(*res):
                    combined_res.append(list(np.concatenate(workers) for workers in zip(*list(layer))))
                return combined_res
            else:
                return list(np.concatenate(worker) for worker in zip(*res))

    def predict(self,x,method='mean_var',full_layer=False,sample_size=50,m=50,aggregation=True):
        """Implement predictions from the trained DGP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            full_layer (bool, optional): whether to output the predictions of all layers. Defaults to `False`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                 Defaults to `50`.
            m (int, optional): the size of the conditioning set for predictions if the DGP was built under the Vecchia approximation. Defaults to `50`.
            aggregation (bool, optional): whether to aggregate mean and variance predictions from imputed linked GPs
                when **method** = '`mean_var`' and **full_layer** = `False`. Defaults to `True`.
            
        Returns:
            tuple_or_list: 
            if the argument **method** = '`mean_var`', a tuple is returned:

                1. If **full_layer** = `False` and **aggregation** = `True`, the tuple contains two numpy 2d-arrays, one for the predictive means 
                   and another for the predictive variances. Each array has its rows corresponding to testing 
                   positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer);
                2. If **full_layer** = `False` and **aggregation** = `False`, the tuple contains two lists, one for the predictive means 
                   and another for the predictive variances from the imputed linked GPs. Each list contains *N* (i.e., the number of imputations) 
                   numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns corresponding to DGP output dimensions 
                   (i.e., the number of GP/likelihood nodes in the final layer);
                3. If **full_layer** = `True`, the tuple contains two lists, one for the predictive means 
                   and another for the predictive variances. Each list contains *L* (i.e., the number of layers) 
                   numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns 
                   corresponding to output dimensions (i.e., the number of GP nodes from the associated layer and in case of the final layer, 
                   it may be the number of the likelihood nodes).

            if the argument **method** = '`sampling`', a list is returned:
                
                1. If **full_layer** = `False`, the list contains *D* (i.e., the number of GP/likelihood nodes in the final layer) numpy 
                   2d-arrays. Each array has its rows corresponding to testing positions and columns corresponding to samples of
                   size: **N** * **sample_size**;
                2. If **full_layer** = `True`, the list contains *L* (i.e., the number of layers) sub-lists. Each sub-list 
                   represents samples drawn from the GPs/likelihoods in the corresponding layers, and contains 
                   *D* (i.e., the number of GP nodes in the corresponding layer or likelihood nodes in the final layer) 
                   numpy 2d-arrays. Each array gives samples of the output from one of *D* GPs/likelihoods at the 
                   testing positions, and has its rows corresponding to testing positions and columns corresponding to samples
                   of size: **N** * **sample_size**.
        """
        if x.ndim==1:
            raise Exception('The testing input has to be a numpy 2d-array')
        if self.all_layer[-1][0].name=='Categorical':
            raise Exception('Use `classify` method to make predictions for the catagorical likelihood.' )
        M=len(x)
        if method=='mean_var':
            sample_size=1
        #start predictions
        mean_pred=[]
        variance_pred=[]
        likelihood_mean=[]
        likelihood_variance=[]
        for s in range(len(self.all_layer_set)):
            overall_global_test_input=x
            one_imputed_all_layer=self.all_layer_set[s]
            if full_layer:
                mean_pred_oneN=[]
                variance_pred_oneN=[]
            for l in range(self.n_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                if l==self.n_layer-1:
                    likelihood_gp_mean=np.empty((M,n_kerenl))
                    likelihood_gp_var=np.empty((M,n_kerenl))
                else:
                    overall_test_output_mean=np.empty((M,n_kerenl))
                    overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                    if full_layer:
                        mean_pred_oneN.append(overall_test_input_mean)
                        variance_pred_oneN.append(overall_test_input_var)
                elif l==self.n_layer-1:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.type=='gp':
                            kernel.pred_m = m
                            if kernel.connect is not None:
                                z_k_in=overall_global_test_input[:,kernel.connect]
                            else:
                                z_k_in=None
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                            likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                        elif kernel.type=='likelihood':
                            m_k,v_k=kernel.prediction(m=m_k_in,v=v_k_in)
                            likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                    if full_layer:
                        mean_pred_oneN.append(overall_test_input_mean)
                        variance_pred_oneN.append(overall_test_input_var)
            for _ in range(sample_size):
                if full_layer:
                    mean_pred.append(mean_pred_oneN)
                    variance_pred.append(variance_pred_oneN)
                else:
                    mean_pred.append(overall_test_input_mean)
                    variance_pred.append(overall_test_input_var)
                likelihood_mean.append(likelihood_gp_mean)
                likelihood_variance.append(likelihood_gp_var)
        if method=='sampling':
            if full_layer:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                samples=[]
                for l in range(self.n_layer):
                    samples_layerwise=[]
                    if l==self.n_layer-1:
                        for mu_likelihood, sigma2_likelihood, dgp_sample in zip(likelihood_mean, likelihood_variance, samples_layer_before_likelihood):
                            realisation=np.empty_like(mu_likelihood)
                            for count, kernel in enumerate(self.all_layer[-1]):
                                if kernel.type=='gp':
                                    realisation[:,count]=np.random.normal(mu_likelihood[:,count],np.sqrt(sigma2_likelihood[:,count]))
                                elif kernel.type=='likelihood':
                                    realisation[:,count]=kernel.sampling(dgp_sample[:,kernel.input_dim])
                            samples_layerwise.append(realisation)
                    else:
                        for mu, sigma2 in zip(mu_layerwise[l], var_layerwise[l]):
                            realisation=np.random.normal(mu,np.sqrt(sigma2))
                            samples_layerwise.append(realisation)
                        if l==self.n_layer-2:
                            samples_layer_before_likelihood=samples_layerwise
                    samples_layerwise=np.asarray(samples_layerwise).transpose(2,1,0)
                    samples.append(list(samples_layerwise))
            else:
                samples=[]
                for mu_dgp, sigma2_dgp, mu_likelihood, sigma2_likelihood  in zip(mean_pred, variance_pred, likelihood_mean, likelihood_variance):
                    realisation=np.empty_like(mu_likelihood)
                    for count, kernel in enumerate(self.all_layer[-1]):
                        if kernel.type=='gp':
                            realisation[:,count]=np.random.normal(mu_likelihood[:,count],np.sqrt(sigma2_likelihood[:,count]))
                        elif kernel.type=='likelihood':
                            dgp_sample=np.random.normal(mu_dgp,np.sqrt(sigma2_dgp))
                            realisation[:,count]=kernel.sampling(dgp_sample[:,kernel.input_dim])
                    samples.append(realisation)
                samples=list(np.asarray(samples).transpose(2,1,0))
            return samples
        elif method=='mean_var':
            if full_layer:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                mu=[np.mean(mu_l,axis=0) for mu_l in mu_layerwise]
                mu2_mean=[np.mean(np.square(mu_l),axis=0) for mu_l in mu_layerwise]
                var_mean=[np.mean(var_l,axis=0) for var_l in var_layerwise]
                sigma2=[i+j-k**2 for i,j,k in zip(mu2_mean,var_mean,mu)]
                mu.append(np.mean(likelihood_mean,axis=0))
                sigma2.append(np.mean((np.square(likelihood_mean)+likelihood_variance),axis=0)-np.mean(likelihood_mean,axis=0)**2)
            else:
                if aggregation:
                    mu=np.mean(likelihood_mean,axis=0)
                    sigma2=np.mean((np.square(likelihood_mean)+likelihood_variance),axis=0)-mu**2
                else:
                    mu=likelihood_mean
                    sigma2=likelihood_variance
            return mu, sigma2
    
    def pclassify(self, x, mode = 'prob', method = 'sampling', full_layer=False, sample_size=50, m=50, chunk_num=None, core_num=None):
        """Implement parallel classification from the trained DGP model with a categorical likelihood.

        Args:
            x, mode, method, full_layer, sample_size, m: see descriptions of the method :meth:`.emulator.classify`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of processes to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``max physical cores available // 2``.

        Returns:
            Same as the method :meth:`.emulator.classify`.
        """
        if self.all_layer[-1][0].name != 'Categorical':
            raise Exception('`pclassify` method is only applicable for DGP models with a catagorical likelihood.')
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
            x_chunk, mode, method, full_layer, sample_size, m = params
            set_num_threads(num_thread)
            return self.classify(x_chunk, mode, method, full_layer, sample_size, m)
        z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            #pool.restart()
            res = pool.map(f, [[x, mode, method, full_layer, sample_size, m] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        if full_layer:
            combined_res=[]
            for layer in zip(*res):
                combined_res.append(list(np.concatenate(workers) for workers in zip(*list(layer))))
            if method == 'mean_var':
                return tuple(combined_res)
            elif method == 'sampling':
                return combined_res
        else:
            return list(np.concatenate(worker) for worker in zip(*res))
    
    def classify(self, x, mode = 'prob', method='sampling', full_layer=False, sample_size=50, m=50):
        """Implement sampling-based classification from the trained DGP model with a categorical likelihood.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            mode (str, optional): whether to generate samples of probabilities of classes (`prob`) or the classes themselves (`label`). Defaults to `prob`.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach when full_layer=True. Defaults to `sampling`.
            full_layer (bool, optional): whether to output the predictions of all layers. Defaults to `False`.
            sample_size (int, optional): the number of samples to draw for each given imputation.
                 Defaults to `50`.
            m (int, optional): the size of the conditioning set for predictions if the DGP was built under the Vecchia approximation. Defaults to `50`.
            
        Returns:
            tuple_or_list: 
                1. If **full_layer** = `False`, the output is a list of *D* numpy 2d-arrays. *D* equals the number of classes if **mode** = `prob` or one
                   if **mode** = `label`. If **mode** = `prob`, each array represents a class and has dimensions where: Rows correspond to the testing positions;
                   Columns correspond to the sampled probabilities (of size **N** * **sample_size**) for that class. If **mode** = `label`, the single 2d-array 
                   in the list represents the sampled class labels (of size **N** * **sample_size**).
                2. If **full_layer** = `True` and **method** = `sampling`, the output is a list containing *L* (i.e., the number of layers) sub-lists. Each of the first *L-1* sub-list 
                   represents samples drawn from GPs in each of the first *L-1* layers. Within each sub-list, there are
                   *D* numpy 2d-arrays, where *D* is the number of GP nodes in that layer. Each 2d-array contains sampled outputs for one of the *D* GPs at 
                   different testing positions. The rows correspond to testing positions, and columns corresponding to samples of size: **N** * **sample_size**.
                   The final sub-list contains either sampled probabilities of each class (when **mode** = `prob`) or sampled class labels (when **mode** = `label`), 
                   as described above when **full_layer** = `False`.
                3. If **full_layer** = `True` and **method** = `mean_var`, the tuple contains three lists, the first list for the predictive means 
                   and the second list for the predictive variances. Each of the two lists contains *L-1* (i.e., the number of first *L-1* layers) 
                   numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns 
                   corresponding to output dimensions (i.e., the number of GP nodes from the associated layer). The final list is a list of *D* numpy 2d-arrays. *D* equals the number of classes if **mode** = `prob` or one
                   if **mode** = `label`. If **mode** = `prob`, each array represents a class and has dimensions where: Rows correspond to the testing positions;
                   Columns correspond to the sampled probabilities (of size **N** * **sample_size**) for that class. If **mode** = `label`, the single 2d-array 
                   in the list represents the sampled class labels (of size **N** * **sample_size**).
        """
        if self.all_layer[-1][0].name != 'Categorical':
            raise Exception('`classify` method is only applicable for DGP models with a catagorical likelihood.' )
        if x.ndim==1:
            raise Exception('The testing input has to be a numpy 2d-array')
        M=len(x)
        if method=='mean_var' and full_layer:
            sample_size_used=1
        else:
            sample_size_used=sample_size
        #start predictions
        mean_pred=[]
        variance_pred=[]
        for s in range(len(self.all_layer_set)):
            overall_global_test_input=x
            one_imputed_all_layer=self.all_layer_set[s]
            if full_layer:
                mean_pred_oneN=[]
                variance_pred_oneN=[]
            for l in range(self.n_layer-1):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                    if full_layer:
                        mean_pred_oneN.append(overall_test_input_mean)
                        variance_pred_oneN.append(overall_test_input_var)
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                    if full_layer:
                        mean_pred_oneN.append(overall_test_input_mean)
                        variance_pred_oneN.append(overall_test_input_var)
            for _ in range(sample_size_used):
                if full_layer:
                    mean_pred.append(mean_pred_oneN)
                    variance_pred.append(variance_pred_oneN)
                else:
                    mean_pred.append(overall_test_input_mean)
                    variance_pred.append(overall_test_input_var)
        if full_layer:
            if method=='sampling':
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                samples=[]
                for l in range(self.n_layer):
                    samples_layerwise=[]
                    if l==self.n_layer-1:
                        for dgp_sample in samples_layer_before_likelihood:
                            realisation=self.all_layer[-1][0].sampling(dgp_sample[:,self.all_layer[-1][0].input_dim], mode=mode)
                            samples_layerwise.append(realisation)
                    else:
                        for mu, sigma2 in zip(mu_layerwise[l], var_layerwise[l]):
                            realisation=np.random.normal(mu,np.sqrt(sigma2))
                            samples_layerwise.append(realisation)
                        if l==self.n_layer-2:
                            samples_layer_before_likelihood=samples_layerwise
                    samples_layerwise=np.asarray(samples_layerwise).transpose(2,1,0)
                    samples.append(list(samples_layerwise))
                return samples
            else:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                mu=[np.mean(mu_l,axis=0) for mu_l in mu_layerwise]
                mu2_mean=[np.mean(np.square(mu_l),axis=0) for mu_l in mu_layerwise]
                var_mean=[np.mean(var_l,axis=0) for var_l in var_layerwise]
                sigma2=[i+j-k**2 for i,j,k in zip(mu2_mean,var_mean,mu)]
                samples=[]
                for mu0, sigma20 in zip(mu_layerwise[self.n_layer-2], var_layerwise[self.n_layer-2]):
                    for _ in range(sample_size):
                        dgp_sample=np.random.normal(mu0,np.sqrt(sigma20))
                        realisation=self.all_layer[-1][0].sampling(dgp_sample[:,self.all_layer[-1][0].input_dim], mode=mode)
                        samples.append(realisation)
                samples=list(np.asarray(samples).transpose(2,1,0))
                return mu, sigma2, samples      
        else:
            samples=[]
            for mu_dgp, sigma2_dgp in zip(mean_pred, variance_pred):
                dgp_sample=np.random.normal(mu_dgp,np.sqrt(sigma2_dgp))
                realisation=self.all_layer[-1][0].sampling(dgp_sample[:,self.all_layer[-1][0].input_dim], mode=mode)
                samples.append(realisation)
            samples=list(np.asarray(samples).transpose(2,1,0))
            return samples
        
    def nllik(self,x,y,m=50):
        """Compute the negative predicted log-likelihood from a trained DGP model with likelihood layer.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            y (ndarray): a numpy 2d-array where each row is a scalar-valued testing output data point.
            m (int, optional): the size of the conditioning set if the DGP was built under the Vecchia approximation. Defaults to `50`.

        Returns:
            tuple: a tuple of two 1d-arrays. The first one is the average negative predicted log-likelihood across
            all testing data points. The second one is the negative predicted log-likelihood for each testing data point.
        """
        if len(self.all_layer[-1])!=1:
            raise Exception('The method is only applicable to a DGP with the final layer formed by only ONE node, which must be a likelihood node.')
        else:
            if self.all_layer[-1][0].type!='likelihood':
                raise Exception('The method is only applicable to a DGP with the final layer formed by only ONE node, which must be a likelihood node.')
        X0, indices = np.unique(x, return_inverse=True, axis=0)
        if len(X0) != len(x):
            x = X0
        M=len(x)
        #start predictions
        predicted_lik=[]
        for s in range(len(self.all_layer_set)):
            overall_global_test_input=x
            one_imputed_all_layer=self.all_layer_set[s]
            for l in range(self.n_layer-1):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        kernel.pred_m = m
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            predicted_lik.append(ghdiag(one_imputed_all_layer[-1][0].pllik,overall_test_input_mean[indices,:],overall_test_input_var[indices,:],y))
        nllik=-np.log(np.mean(predicted_lik,axis=0)).flatten()
        average_nllik=np.mean(nllik)
        return average_nllik, nllik

    def log_loss(self, x, y, sample_size=50, m=50):
        """Compute the log loss from a trained DGP classifier.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            y (ndarray): a numpy 1d-array that gives the testing output labels.
            sample_size (int, optional): the number of samples to draw for each given imputation.
                 Defaults to `50`.
            m (int, optional): the size of the conditioning set for predictions if the DGP was built under the Vecchia approximation. Defaults to `50`.

        Returns:
            a scalar that gives the log loss value.
        """
        if self.all_layer[-1][0].name!='Categorical':
            raise Exception('The method is only applicable to DGPs with categorical likelihoods.')
        x_unique, order = np.unique(x, return_inverse = True, axis = 0)
        prob_samp = self.classify(x = x_unique, mode = 'prob', full_layer=False, sample_size=sample_size, m=m)
        prob_samp = np.asarray(prob_samp).transpose(2,1,0)
        y_encode = self.all_layer[-1][0].class_encoder.transform(y)
        ll = logloss(prob_samp, y_encode, order)
        return(ll)

        
      