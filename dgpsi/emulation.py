import multiprocess.context as ctx
import platform
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil   
from .imputation import imputer
import copy
from .functions import ghdiag, mice_var

class emulator:
    """Class to make predictions from the trained DGP model.

    Args:
        all_layer (list): a list that contains the trained DGP model produced by the method :meth:`.estimate`
            of the :class:`.dgp` class. 
        N (int, optional): the number of imputations to produce the predictions. Increase the value to account for
            more imputation uncertainties. Defaults to `50`.
        nb_parallel (bool, optional): whether to use *Numba*'s multi-threading to accelerate the predictions. Defaults to `False`.
    """
    def __init__(self, all_layer, N=50, nb_parallel=False):
        self.all_layer=all_layer
        self.n_layer=len(all_layer)
        self.imp=imputer(self.all_layer)
        (self.imp).sample(burnin=50)
        self.all_layer_set=[]
        for _ in range(N):
            (self.imp).sample()
            (self.imp).key_stats()
            (self.all_layer_set).append(copy.deepcopy(self.all_layer))
        self.all_layer_set_copy = copy.deepcopy(self.all_layer_set)
        self.nb_parallel=nb_parallel
        #if len(self.all_layer[0][0].input)>=500 and self.nb_parallel==False:
        #    print('Your training data size is greater than %i, you might want to set "nb_parallel=True" to accelerate the prediction.' % (500))
    
    def set_nb_parallel(self,nb_parallel):
        """Set **self.nb_parallel** to the bool value given by **nb_parallel**. This method is useful to change **self.nb_parallel**
            when the :class:`.emulator` class has already been built.
        """
        self.nb_parallel=nb_parallel

    def esloo(self, X, Y):
        """Compute the (normalised) expected squared LOO from a DGP emulator.

        Args:
            X (ndarray): the training input data used to build the DGP emulator via the :class:`.dgp` class.
            Y (ndarray): the training output data used to build the DGP emulator via the :class:`.dgp` class.
            
        Returns:
            ndarray: a numpy 2d-array is returned. The array has its rows corresponding to training input
                positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer);
        """
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        else:
            indices = None
        res = []
        for i in range(len(X)):
            res.append(self.esloo_calculation(i, X, Y, indices))
        self.all_layer_set=copy.deepcopy(self.all_layer_set_copy)
        final_res = np.concatenate(res)
        if isrep:
            idx=[]
            seq=np.arange(len(Y))
            for i in range(len(X)):
                idx.append(seq[indices==i])
            idx=np.concatenate(idx)
            final_res_cp=np.empty_like(final_res)
            final_res_cp[idx,:]=final_res
            return final_res_cp
        else:
            return final_res

    def pesloo(self, X, Y, core_num=None):
        """Compute in parallel the (normalised) expected squared LOO from a DGP emulator.

        Args:
            X, Y: see descriptions of the method :meth:`.emulator.esloo`.
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.emulator.esloo`.
        """
        if platform.system()=='Darwin':
            ctx._force_start_method('forkserver')
        if core_num is None:
            core_num=psutil.cpu_count(logical = False)-1
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        else:
            indices = None
        f=lambda x: self.esloo_calculation(*x) 
        z=list(np.arange(len(X)))
        with Pool(core_num) as pool:
            #pool.restart()
            res = pool.map(f, [[x, X, Y, indices] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        self.all_layer_set=copy.deepcopy(self.all_layer_set_copy)
        final_res = np.concatenate(res)
        if isrep:
            idx=[]
            seq=np.arange(len(Y))
            for i in range(len(X)):
                idx.append(seq[indices==i])
            idx=np.concatenate(idx)
            final_res_cp=np.empty_like(final_res)
            final_res_cp[idx,:]=final_res
            return final_res_cp
        else:
            return final_res

    def esloo_calculation(self, i, X, Y, indices):
        for s in range(len(self.all_layer_set)):
            one_imputed_all_layer=self.all_layer_set[s]
            for l in range(self.n_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                for k in range(n_kerenl):
                    kernel=layer[k]
                    kernel_ref=self.all_layer_set_copy[s][l][k]
                    if kernel_ref.rep is None:
                        kernel.input = np.delete(kernel_ref.input, i, 0)
                        kernel.output = np.delete(kernel_ref.output, i, 0)
                        if kernel.type == 'gp':
                            if kernel_ref.global_input is not None:
                                kernel.global_input = np.delete(kernel_ref.global_input, i, 0)
                            kernel.Rinv = kernel_ref.cv_stats(i)
                            kernel.Rinv_y=np.dot(kernel.Rinv,kernel.output).flatten()
                            if kernel.name=='sexp':
                                kernel.R2sexp = np.delete(np.delete(kernel_ref.R2sexp, i, 0), i, 1)
                                kernel.Psexp = np.delete(np.delete(kernel_ref.Psexp, i, 1), i, 2)
                    else:
                        idx = kernel_ref.rep!=i
                        kernel.input = (kernel_ref.input[idx,:]).copy()
                        kernel.output = (kernel_ref.output[idx,:]).copy()
                        if kernel.type == 'gp':
                            if kernel_ref.global_input is not None:
                                kernel.global_input = (kernel_ref.global_input[idx,:]).copy()
                            kernel.Rinv = kernel_ref.cv_stats(i)
                            kernel.Rinv_y=np.dot(kernel.Rinv,kernel.output).flatten()
                            if kernel.name=='sexp':
                                kernel.R2sexp = (kernel_ref.R2sexp[idx,:][:,idx]).copy()
                                kernel.Psexp = (kernel_ref.Psexp[:,idx,:][:,:,idx]).copy()
        mu_i, var_i = self.predict(x=X[[i],:], aggregation=False)
        mu=np.mean(mu_i,axis=0)
        sigma2=np.mean((np.square(mu_i)+var_i),axis=0)-mu**2
        if indices is not None:
            f=Y[indices==i,:]
        else:
            f=Y[[i],:]
        esloo=sigma2+(mu-f)**2
        error=(mu_i-f)**2
        normaliser=np.mean(error**2+6*error*var_i+3*np.square(var_i),axis=0)-esloo**2
        nesloo=esloo/np.sqrt(normaliser)
        return(nesloo)
    
    def loo(self, X, method='mean_var', sample_size=50):
        """Implement the Leave-One-Out cross-validation from a DGP emulator.

        Args:
            X (ndarray): the training input data used to build the DGP emulator via the :class:`.dgp` class.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach for the LOO. Defaults to `mean_var`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                 Defaults to `50`.
            
        Returns:
            tuple_or_list: 
            if the argument **method** = '`mean_var`', a tuple is returned. The tuple contains two numpy 2d-arrays, one for the predictive means 
                and another for the predictive variances. Each array has its rows corresponding to training input
                positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer);

            if the argument **method** = '`sampling`', a list is returned. The list contains *D* (i.e., the number of GP/likelihood nodes in the 
                final layer) numpy 2d-arrays. Each array has its rows corresponding to training input positions and columns corresponding to samples 
                of size: **N** * **sample_size**;
        """
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        res = []
        for i in range(len(X)):
            res.append(self.loo_calculation(i, X, method, sample_size))
        self.all_layer_set=copy.deepcopy(self.all_layer_set_copy)
        final_res = list(np.concatenate(res_i) for res_i in zip(*res)) 
        if isrep:
            for j in range(len(final_res)):
                final_res[j] = final_res[j][indices,:]
        if method == 'mean_var':
            return tuple(final_res)
        elif method == 'sampling':
            return final_res

    def ploo(self, X, method='mean_var', sample_size=50, core_num=None):
        """Implement the parallel Leave-One-Out cross-validation from a DGP emulator.

        Args:
            X, method, sample_size: see descriptions of the method :meth:`.emulator.loo`.
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.emulator.loo`.
        """
        if platform.system()=='Darwin':
            ctx._force_start_method('forkserver')
        if core_num is None:
            core_num=psutil.cpu_count(logical = False)-1
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        f=lambda x: self.loo_calculation(*x) 
        z=list(np.arange(len(X)))
        with Pool(core_num) as pool:
            #pool.restart()
            res = pool.map(f, [[x, X, method, sample_size] for x in z])
            pool.close()
            pool.join()
            pool.clear()
        self.all_layer_set=copy.deepcopy(self.all_layer_set_copy)
        final_res = list(np.concatenate(worker) for worker in zip(*res)) 
        if isrep:
            for j in range(len(final_res)):
                final_res[j] = final_res[j][indices,:]
        if method == 'mean_var':
            return tuple(final_res)
        elif method == 'sampling':
            return final_res
            
    def loo_calculation(self, i, X, method, sample_size):
        for s in range(len(self.all_layer_set)):
            one_imputed_all_layer=self.all_layer_set[s]
            for l in range(self.n_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                for k in range(n_kerenl):
                    kernel=layer[k]
                    kernel_ref=self.all_layer_set_copy[s][l][k]
                    if kernel_ref.rep is None:
                        kernel.input = np.delete(kernel_ref.input, i, 0)
                        kernel.output = np.delete(kernel_ref.output, i, 0)
                        if kernel.type == 'gp':
                            if kernel_ref.global_input is not None:
                                kernel.global_input = np.delete(kernel_ref.global_input, i, 0)
                            kernel.Rinv = kernel_ref.cv_stats(i)
                            kernel.Rinv_y=np.dot(kernel.Rinv,kernel.output).flatten()
                            if kernel.name=='sexp':
                                kernel.R2sexp = np.delete(np.delete(kernel_ref.R2sexp, i, 0), i, 1)
                                kernel.Psexp = np.delete(np.delete(kernel_ref.Psexp, i, 1), i, 2)
                    else:
                        idx = kernel_ref.rep!=i
                        kernel.input = (kernel_ref.input[idx,:]).copy()
                        kernel.output = (kernel_ref.output[idx,:]).copy()
                        if kernel.type == 'gp':
                            if kernel_ref.global_input is not None:
                                kernel.global_input = (kernel_ref.global_input[idx,:]).copy()
                            kernel.Rinv = kernel_ref.cv_stats(i)
                            kernel.Rinv_y=np.dot(kernel.Rinv,kernel.output).flatten()
                            if kernel.name=='sexp':
                                kernel.R2sexp = (kernel_ref.R2sexp[idx,:][:,idx]).copy()
                                kernel.Psexp = (kernel_ref.Psexp[:,idx,:][:,:,idx]).copy()
        res = self.predict(x=X[[i],:], method=method, sample_size=sample_size)
        return(res)

    def pmetric(self, x_cand, method='ALM',nugget_s=1.,score_only=False,chunk_num=None,core_num=None):
        """Compute the value of the ALM or MICE criterion for sequential designs in parallel.

        Args:
            x_cand, method, nugget_s, score_only: see descriptions of the method :meth:`.emulator.metric`.
            chunk_num (int, optional): the number of chunks that the candidate design set **x_cand** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.emulator.metric`.
        """
        if x_cand.ndim==1:
            raise Exception('The candidate design set has to be a numpy 2d-array.')
        if self.all_layer[self.n_layer-1][0].type=='likelihood':
            raise Exception('The method is only applicable to DGPs without likelihood layers.')
        if method == 'ALM':
            _, sigma2 = self.ppredict(x=x_cand,chunk_num=chunk_num,core_num=core_num)
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,np.arange(sigma2.shape[1])]
        elif method == 'MICE':
            if platform.system()=='Darwin':
                ctx._force_start_method('forkserver')
            if core_num is None:
                core_num=psutil.cpu_count(logical = False)-1
            if chunk_num is None:
                chunk_num=core_num
            if chunk_num<core_num:
                core_num=chunk_num
            f=lambda x: self.predict_mice(*x) 
            z=np.array_split(x_cand,chunk_num)
            with Pool(core_num) as pool:
                res = pool.map(f, [[x] for x in z])
                pool.close()
                pool.join()
                pool.clear()
            combined_res=[]
            for element in zip(*res):
                combined_res.append(list(np.concatenate(workers) for workers in zip(*list(element))))
            predicted_input, sigma2 = combined_res[0], combined_res[1]   
            M=len(x_cand)
            D=len(self.all_layer[-1])
            mice=np.zeros((M,D))
            S=len(self.all_layer_set)
            for i in range(S):
                last_layer=self.all_layer_set[i][-1]
                sigma2_s_i=np.empty((M,D))
                for k in range(D):
                    kernel = last_layer[k]
                    sigma2_s_i[:,k] = mice_var(predicted_input[i], x_cand, copy.deepcopy(kernel), nugget_s).flatten()
                with np.errstate(divide='ignore'):
                    mice =+ np.log(sigma2[i]/sigma2_s_i)
            avg_mice=mice/S
            if score_only:
                return avg_mice
            else:
                idx = np.argmax(avg_mice, axis=0)
                return idx, avg_mice[idx,np.arange(avg_mice.shape[1])]

    def metric(self, x_cand, method='ALM',nugget_s=1.,score_only=False):
        """Compute the value of the ALM or MICE criterion for sequential designs.

        Args:
            x_cand (ndarray): a numpy 2d-array that represents a candidate input design where each row is a design point and 
                each column is a design input dimension.
            method (str, optional): the sequential design approach: MICE (`MICE`) or ALM 
                (`ALM`). Defaults to `ALM`.
            nugget_s (float, optional): the value of the smoothing nugget term used when **method** = '`MICE`'. Defaults to `1.0`.
            score_only (bool, optional): whether to return only the scores of ALM or MICE criterion at all design points contained in **x_cand**.
                Defaults to `False`.

        Returns:
            ndarray_or_tuple: 
            if the argument **score_only** = `True`, a numpy 2d-array is returned that gives the scores of ALM or MICE criterion with rows
               corresponding to design points in the candidate design set **x_cand** and columns corresponding to output dimensions;

            if the argument **score_only** = `False`, a tuple of two numpy 1d-arrays is returned. The first one gives the indices (i.e., row numbers) 
                of the design points in the candidate design set **x_cand** that have the largest criterion values, which are given by the second array, 
                across different outputs of the DGP emulator.
        """
        if x_cand.ndim==1:
            raise Exception('The candidate design set has to be a numpy 2d-array.')
        if self.all_layer[self.n_layer-1][0].type=='likelihood':
            raise Exception('The method is only applicable to DGPs without likelihood layers.')
        if method == 'ALM':
            _, sigma2 = self.predict(x=x_cand)
            if score_only:
                return sigma2
            else:
                idx = np.argmax(sigma2, axis=0)
                return idx, sigma2[idx,np.arange(sigma2.shape[1])]
        elif method == 'MICE':
            predicted_input, sigma2 = self.predict_mice(x_cand)
            M=len(x_cand)
            D=len(self.all_layer[-1])
            mice=np.zeros((M,D))
            S=len(self.all_layer_set)
            for i in range(S):
                last_layer=self.all_layer_set[i][-1]
                sigma2_s_i=np.empty((M,D))
                for k in range(D):
                    kernel = last_layer[k]
                    sigma2_s_i[:,k] = mice_var(predicted_input[i], x_cand, copy.deepcopy(kernel), nugget_s).flatten()
                with np.errstate(divide='ignore'):
                    mice =+ np.log(sigma2[i]/sigma2_s_i)
            avg_mice=mice/S
            if score_only:
                return avg_mice
            else:
                idx = np.argmax(avg_mice, axis=0)
                return idx, avg_mice[idx,np.arange(avg_mice.shape[1])]
            
    def predict_mice(self,x_cand):
        """Implement predictions from the trained DGP model that are required to calculate the MICE criterion.
        """
        S=len(self.all_layer_set)
        M=len(x_cand)
        D=len(self.all_layer[-1])
        variance_pred_set=[]
        pred_input_set=[]
        #start calculation
        for i in range(S):
            one_imputed_all_layer=self.all_layer_set[i]
            variance_pred=np.empty((M,D))
            overall_global_test_input=x_cand
            for l in range(self.n_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
                elif l==self.n_layer-1:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        _,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in,nb_parallel=self.nb_parallel)
                        variance_pred[:,k]=v_k
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in,nb_parallel=self.nb_parallel)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            variance_pred_set.append(variance_pred)
            pred_input_set.append(overall_test_input_mean)
        return pred_input_set, variance_pred_set

    def ppredict(self,x,method='mean_var',full_layer=False,sample_size=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained DGP model.

        Args:
            x, method, full_layer, sample_size: see descriptions of the method :meth:`.emulator.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method :meth:`.emulator.predict`.
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
            #pool.restart()
            res = pool.map(f, [[x, method, full_layer, sample_size, True] for x in z])
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

    def predict(self,x,method='mean_var',full_layer=False,sample_size=50,aggregation=True):
        """Implement predictions from the trained DGP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            full_layer (bool, optional): whether to output the predictions of all layers. Defaults to `False`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                 Defaults to `50`.
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
                            if kernel.connect is not None:
                                z_k_in=overall_global_test_input[:,kernel.connect]
                            else:
                                z_k_in=None
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in,nb_parallel=self.nb_parallel)
                            likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                        elif kernel.type=='likelihood':
                            m_k,v_k=kernel.prediction(m=m_k_in,v=v_k_in)
                            likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in,nb_parallel=self.nb_parallel)
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

    def nllik(self,x,y):
        """Compute the negative predicted log-likelihood from a trained DGP model with likelihood layer.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            y (ndarray): a numpy 2d-array where each row is a scalar-valued testing output data point.

        Returns:
            tuple: a tuple of two 1d-arrays. The first one is the average negative predicted log-likelihood across
            all testing data points. The second one is the negative predicted log-likelihood for each testing data point.
        """
        if len(self.all_layer[-1])!=1:
            raise Exception('The method is only applicable to a DGP with the final layer formed by only ONE node, which must be a likelihood node.')
        else:
            if self.all_layer[-1][0].type!='likelihood':
                raise Exception('The method is only applicable to a DGP with the final layer formed by only ONE node, which must be a likelihood node.')
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
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if kernel.connect is not None:
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in,nb_parallel=self.nb_parallel)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                    overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            predicted_lik.append(ghdiag(one_imputed_all_layer[-1][0].pllik,overall_test_input_mean,overall_test_input_var,y))
        nllik=-np.log(np.mean(predicted_lik,axis=0)).flatten()
        average_nllik=np.mean(nllik)
        return average_nllik, nllik

        
      