import numpy as np
from .imputation import imputer
import copy
from scipy.spatial.distance import cdist
from .functions import ghdiag, mice_var
from .vecchia import get_pred_nn
from contextlib import contextmanager

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
        self.all_layer_set=[None] * N
        for i in range(N):
            if self.vecch:
                (self.imp).update_ord_nn()
            (self.imp).sample()
            if not self.vecch:
                (self.imp).key_stats()
            self.all_layer_set[i] = copy.deepcopy(self.all_layer)
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
        
    def loo(self, X, method=None, sample_size=50, m=30):
        """Implement the Leave-One-Out cross-validation from a DGP emulator.

        Args:
            X (ndarray): the training input data used to build the DGP emulator via the :class:`.dgp` class.
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach for the LOO. If set to None, 
                mean-variance (`mean_var`) approach is used. Defaults to None.
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
            method = 'mean_var'
        isrep = len(X) != len(self.all_layer[0][0].input)
        if isrep:
            X, indices = np.unique(X, return_inverse=True, axis=0)
        m_pred = m+1 if self.vecch else X.shape[0]
        with self.change_vecch_state():
            final_res = self.predict(X, method=method, sample_size=sample_size, m=m_pred)
        if isrep:
            modified_items = [item[indices, :] for item in final_res]
            final_res = type(final_res)(modified_items)
        return final_res

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
                   positions and columns corresponding to DGP output dimensions (i.e., the number of GP/likelihood nodes in the final layer). 
                   For categorical likelihood, the arrays represent the predictive means and variances of class probabilities. The number of 
                   columns corresponds to the number of classes. In the binary classification case, a single column is returned, representing 
                   the probabilities of class 1;
                2. If **full_layer** = `False` and **aggregation** = `False`, the tuple contains two lists, one for the predictive means 
                   and another for the predictive variances from the imputed linked GPs. Each list contains *N* (i.e., the number of imputations) 
                   numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns corresponding to DGP output dimensions 
                   (i.e., the number of GP/likelihood nodes in the final layer). For categorical likelihood, arrays in each list represent 
                   the predictive means and variances of class probabilities obtained from the imputed linked GPs. The number of columns of the 
                   arrays corresponds to the number of classes. In the binary classification case, arrays have a single column, representing the probabilities of class 1;
                3. If **full_layer** = `True`, the tuple contains two lists, one for the predictive means 
                   and another for the predictive variances. Each list contains *L* (i.e., the number of layers) 
                   numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns 
                   corresponding to output dimensions (i.e., the number of GP nodes from the associated layer and in case of the final layer, 
                   it may be the number of the likelihood nodes). For categorical likelihood, the final arrays in each list represent 
                   the predictive means and variances of class probabilities. The number of columns of the final arrays corresponds to the number of 
                   classes. In the binary classification case, a single column is returned, representing the probabilities of class 1;

            if the argument **method** = '`sampling`', a list is returned:
                
                1. If **full_layer** = `False`, the list contains *D* (i.e., the number of GP/likelihood nodes in the final layer) numpy 
                   2d-arrays. Each array has its rows corresponding to testing positions and columns corresponding to samples of
                   size: **N** * **sample_size**. For the categorical likelihood, the list contains numpy 2d-arrays of class probabilities, 
                   with the number of arrays equal to the number of classes. In the binary classification case, only a single array is returned, 
                   representing samples of the probability of class 1;
                2. If **full_layer** = `True`, the list contains *L* (i.e., the number of layers) sub-lists. Each sub-list 
                   represents samples drawn from the GPs/likelihoods in the corresponding layers, and contains 
                   *D* (i.e., the number of GP nodes in the corresponding layer or likelihood nodes in the final layer) 
                   numpy 2d-arrays. Each array gives samples of the output from one of *D* GPs/likelihoods at the 
                   testing positions, and has its rows corresponding to testing positions and columns corresponding to samples
                   of size: **N** * **sample_size**.  For the categorical likelihood, the final sub-list contains numpy 2d-arrays of class probabilities, 
                   with the number of arrays equal to the number of classes. In the binary classification case, the final sub-set only contains a single array, 
                   representing samples of the probability of class 1.
        """
        if x.ndim != 2:
            raise Exception('The testing input has to be a numpy 2d-array')
        is_cat = False
        if self.all_layer[-1][0].name=='Categorical':
            is_cat = True
            n_class = self.all_layer[-1][0].num_classes
        #   raise Exception('Use `classify` method to make predictions for the catagorical likelihood.' )
        M=len(x)
        S = len(self.all_layer_set)
        T = S * sample_size

        do_stream_agg = (method == 'mean_var') and (not full_layer) and aggregation
        if do_stream_agg:
            likelihood_mean_sum = None
            likelihood_m2v_sum = None

        #start predictions
        mean_pred=[]
        variance_pred=[]
        likelihood_mean=[]
        likelihood_variance=[]

        if full_layer:
            mean_pred_layers = []      # list length S, each is list of (n_layer-1) arrays
            variance_pred_layers = []

        for s in range(S):
            overall_global_test_input=x
            one_imputed_all_layer=self.all_layer_set[s]
            if full_layer:
                mean_pred_oneN=[]
                variance_pred_oneN=[]
            for l in range(self.n_layer):
                layer=one_imputed_all_layer[l]
                n_kerenl=len(layer)
                if l==self.n_layer-1:
                    if is_cat:
                        if n_class==2:
                            likelihood_gp_mean=np.empty((M,1))
                            likelihood_gp_var=np.empty((M,1))
                        else:
                            likelihood_gp_mean=np.empty((M,n_class))
                            likelihood_gp_var=np.empty((M,n_class))
                    else:
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
                            if is_cat:
                                likelihood_gp_mean[:,:], likelihood_gp_var[:,:] = m_k_in, v_k_in
                            else:  
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
            
            if full_layer:
                mean_pred_layers.append(mean_pred_oneN)
                variance_pred_layers.append(variance_pred_oneN)

            if (method == 'sampling') and (not full_layer):
                mean_pred.append(overall_test_input_mean)
                variance_pred.append(overall_test_input_var)

            # For mean_var+aggregation+full_layer=False, stream aggregate and do not store lists
            if do_stream_agg:
                if likelihood_mean_sum is None:
                    likelihood_mean_sum = likelihood_gp_mean.copy()
                    likelihood_m2v_sum = (likelihood_gp_mean * likelihood_gp_mean + likelihood_gp_var)
                else:
                    likelihood_mean_sum += likelihood_gp_mean
                    likelihood_m2v_sum += (likelihood_gp_mean * likelihood_gp_mean + likelihood_gp_var)
            else:
                likelihood_mean.append(likelihood_gp_mean)
                likelihood_variance.append(likelihood_gp_var)

        if do_stream_agg:
            agg_mean = likelihood_mean_sum / S
            agg_var = (likelihood_m2v_sum / S) - agg_mean * agg_mean
            if is_cat:
                mu, sigma2 = self.all_layer[-1][0].prediction(agg_mean, agg_var)
                return mu, sigma2
            else:
                return agg_mean, agg_var

        if method=='sampling':
            if full_layer:
                samples = []

                samples_layer_before_likelihood = None

                for l in range(self.n_layer - 1):
                    D_l = mean_pred_layers[0][l].shape[1]

                    samples_layer_arr = np.empty((T, M, D_l))

                    for s, (mu, sigma2) in enumerate(zip(
                        (item[l] for item in mean_pred_layers),
                        (item[l] for item in variance_pred_layers),
                    )):
                        t0, t1 = s * sample_size, (s + 1) * sample_size

                        block = np.random.normal(
                            loc=mu,
                            scale=np.sqrt(sigma2),
                            size=(sample_size,) + mu.shape
                        )  # (sample_size, M, D_l)

                        samples_layer_arr[t0:t1, :, :] = block

                    if l == self.n_layer - 2:
                        samples_layer_before_likelihood = samples_layer_arr  # shape (T, M, D_prev)

                    # output format: list of D_l arrays, each (M, T)
                    samples.append(list(samples_layer_arr.transpose(2, 1, 0)))

                samples_arr = np.empty((T, M, likelihood_mean[0].shape[1]))

                for s, (mu_likelihood, sigma2_likelihood) in enumerate(zip(likelihood_mean, likelihood_variance)):
                    t0, t1 = s * sample_size, (s + 1) * sample_size

                    block = np.random.normal(
                        loc=mu_likelihood,
                        scale=np.sqrt(sigma2_likelihood),
                        size=(sample_size,) + mu_likelihood.shape
                    )  # (sample_size, M, D)

                    samples_arr[t0:t1, :, :] = block

                lik_nodes = [(count, kernel) for count, kernel in enumerate(self.all_layer[-1]) if kernel.type == 'likelihood']

                if lik_nodes:
                    for t, dgp_sample in enumerate(samples_layer_before_likelihood):
                        for count, kernel in lik_nodes:
                            if is_cat:
                                samples_arr[t, :, :] = kernel.sampling(dgp_sample[:, kernel.input_dim])
                            else:
                                samples_arr[t, :, count] = kernel.sampling(dgp_sample[:, kernel.input_dim])

                samples.append(list(samples_arr.transpose(2, 1, 0)))
            else:
                samples_arr = np.empty((T, M, likelihood_mean[0].shape[1]))
                lik_nodes = [(count, kernel) for count, kernel in enumerate(self.all_layer[-1]) if kernel.type == 'likelihood']

                for s, (mu_dgp, sigma2_dgp, mu_likelihood, sigma2_likelihood) in enumerate(
                    zip(mean_pred, variance_pred, likelihood_mean, likelihood_variance)
                ):
                    t0, t1 = s * sample_size, (s + 1) * sample_size

                    block = np.random.normal(
                        loc=mu_likelihood,
                        scale=np.sqrt(sigma2_likelihood),
                        size=(sample_size,) + mu_likelihood.shape
                    ) 

                    # overwrite likelihood outputs only if present
                    if lik_nodes:
                        for r in range(sample_size):
                            realisation = block[r]  # view (M, D)
                            for count, kernel in lik_nodes:
                                dgp_sample = np.random.normal(mu_dgp, np.sqrt(sigma2_dgp))
                                if is_cat:
                                    realisation[:, :] = kernel.sampling(dgp_sample[:, kernel.input_dim])
                                else:
                                    realisation[:, count] = kernel.sampling(dgp_sample[:, kernel.input_dim])

                    samples_arr[t0:t1, :, :] = block

                samples = list(samples_arr.transpose(2, 1, 0))
            return samples
        elif method=='mean_var':
            if full_layer:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred_layers)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred_layers)]
                mu=[np.mean(mu_l,axis=0) for mu_l in mu_layerwise]
                mu2_mean=[np.mean(np.square(mu_l),axis=0) for mu_l in mu_layerwise]
                var_mean=[np.mean(var_l,axis=0) for var_l in var_layerwise]
                sigma2=[i+j-k**2 for i,j,k in zip(mu2_mean,var_mean,mu)]
                if is_cat:
                    agg_mean, agg_var = np.mean(likelihood_mean,axis=0), np.mean((np.square(likelihood_mean)+likelihood_variance),axis=0)-np.mean(likelihood_mean,axis=0)**2
                    m_agg,v_agg = self.all_layer[-1][0].prediction(m=agg_mean,v=agg_var)
                    mu.append(m_agg)
                    sigma2.append(v_agg)
                else:
                    mu.append(np.mean(likelihood_mean,axis=0))
                    sigma2.append(np.mean((np.square(likelihood_mean)+likelihood_variance),axis=0)-np.mean(likelihood_mean,axis=0)**2)
            else:
                if is_cat:
                    mu, sigma2 = [list(x) for x in zip(*(self.all_layer[-1][0].prediction(a, b) for a, b in zip(likelihood_mean, likelihood_variance)))]
                else:
                    mu=likelihood_mean
                    sigma2=likelihood_variance
            return mu, sigma2
        
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
        S = len(self.all_layer_set)
        N_latent_layers = self.n_layer - 1  # layers before likelihood

        # Precompute kernel counts per latent layer
        latent_widths = [len(self.all_layer_set[0][l]) for l in range(N_latent_layers)]
        mean_bufs = [np.empty((M, w)) for w in latent_widths]
        var_bufs  = [np.empty((M, w)) for w in latent_widths]

        log_sum = None

        for s in range(S):
            one_imputed_all_layer = self.all_layer_set[s]

            for l in range(N_latent_layers):
                layer = one_imputed_all_layer[l]
                out_mean = mean_bufs[l]
                out_var  = var_bufs[l]
                n_kernel = len(layer)

                if l == 0:
                    # inputs come from x
                    for k in range(n_kernel):
                        kernel = layer[k]
                        kernel.pred_m = m
                        z_k_in = x[:, kernel.connect] if kernel.connect is not None else None
                        mk, vk = kernel.gp_prediction(x=x[:, kernel.input_dim], z=z_k_in)
                        out_mean[:, k] = mk
                        out_var[:, k]  = vk
                else:
                    # inputs come from previous layer buffers
                    prev_mean = mean_bufs[l - 1]
                    prev_var  = var_bufs[l - 1]
                    for k in range(n_kernel):
                        kernel = layer[k]
                        kernel.pred_m = m
                        m_in = prev_mean[:, kernel.input_dim]
                        v_in = prev_var[:, kernel.input_dim]
                        z_k_in = x[:, kernel.connect] if kernel.connect is not None else None
                        mk, vk = kernel.linkgp_prediction(m=m_in, v=v_in, z=z_k_in)
                        out_mean[:, k] = mk
                        out_var[:, k]  = vk

            # latent stats at the final latent layer:
            latent_mean = mean_bufs[N_latent_layers - 1]
            latent_var  = var_bufs[N_latent_layers - 1]

            # compute likelihood for original x (incl duplicates) using indices
            p = ghdiag(
                one_imputed_all_layer[-1][0].pllik,
                latent_mean[indices, :],
                latent_var[indices, :],
                y
            ).reshape(-1)

            # accumulate log-sum-exp in a streaming way
            logp = np.log(p)
            log_sum = logp if log_sum is None else np.logaddexp(log_sum, logp)

        # log(mean_s p_s) = logsumexp - log(S)
        log_mean = log_sum - np.log(S)
        nllik = -log_mean
        return float(np.mean(nllik)), nllik

        
      