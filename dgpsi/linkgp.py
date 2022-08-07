import multiprocess.context as ctx
import platform
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil
from .imputation import imputer
import copy

class container:
    """
    Class to contain the trained GP or DGP emulator of a computer model for linked (D)GP emulation.

    Args:
        structure (list): a list that contains the trained structure of GP or DGP of a computer model. For GP, 
            this is the list exported from the :meth:`.export` method of the :class:`.gp` class. For DGP, this is the list exported 
            from the :meth:`.estimate` of the :class:`.dgp` class.
        local_input_idx (ndarray): a numpy 1d-array that specifies the indices of outputs (a 2d-array) 
            produced by all models in the feeding layer that are input to the model emulated by the GP or DGP
            represented by the **structure** argument. The indices should be ordered in such a way that the extracted
            output from the feeding layer is sorted in the same order as the training input used for the GP/DGP 
            emulation of the computer model that the **structure** argument represents. When the model is in the first
            layer, **local_input_idx** gives the indices of its input in the global testing input set, see :meth:`.lgp.predict`
            for descriptions of the global testing input set. Defaults to `None`. When the
            argument is `None`, one needs to set its value using the :meth:`.set_local_input`. 
    """
    def __init__(self, structure, local_input_idx=None):
        if len(structure)==1:
            self.type='gp'
            self.structure=structure[0]
        else:
            self.type='dgp'
            self.structure=structure
            self.imp=imputer(self.structure)
            self.imp.sample(burnin=50)
        self.local_input_idx=local_input_idx

    def set_local_input(self, idx):
        """Set the **local_input_idx** argument if it is not set when the class is constructed.
        """
        self.local_input_idx=idx

class lgp:
    """
    Class to store a system of GP and DGP emulators for predictions. 

    Args:
        all_layer (list): a list contains *L* (the number of layers of a systems of computer models) sub-lists, 
            each of which represents a layer and contains the GP/DGP emulators of computer models represented by 
            the :class:`.container` class. The sub-lists are placed in the list in the same order of the specified computer
            model system.
        N (int): the number of imputation to produce the predictions. Increase the value to account for more 
            imputation uncertainties. If the system consists only GP emulators, **N** is set to `1` automatically. 
            Defaults to `50`.
        nb_parallel (bool, optional): whether to use *Numba*'s multi-threading to accelerate the predictions. Defaults to `False`.
    """
    def __init__(self, all_layer, N=50, nb_parallel=False):
        self.nb_parallel=nb_parallel
        self.L=len(all_layer)
        self.all_layer=all_layer
        self.num_model=[]
        for l in range(1,self.L):
            self.num_model.append(len(all_layer[l]))
        if np.sum([[cont.type=='dgp' for cont in all_layer[l]] for l in range(self.L)])==0:
            N=1
        self.all_layer_set=[]
        for _ in range(N):
            one_imputation=[]
            for l in range(self.L):
                layer=[]
                for cont in all_layer[l]:
                    if cont.type=='gp':
                        layer.append(copy.deepcopy(cont))
                    elif cont.type=='dgp':
                        (cont.imp).sample()
                        (cont.imp).key_stats()
                        layer.append(copy.deepcopy(cont))
                one_imputation.append(layer)
            self.all_layer_set.append(one_imputation)

    def set_nb_parallel(self,nb_parallel):
        """Set 'self.nb_parallel' to the bool value given by **nb_parallel**. This method is useful to change **self.nb_parallel**
            when the :class:`.lgp` class has already been built.
        """
        self.nb_parallel=nb_parallel
    
    def ppredict(self,x,method='mean_var',full_layer=False,sample_size=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained DGP model.

        Args:
            x, method, full_layer, sample_size: see descriptions of the method :meth:`.lgp.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method `predict`.
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
        if isinstance(x, list):
            if len(x)!=self.L:
                raise Exception('When test input is given as a list, it must contain global inputs to the all layers (even with no global inputs to internal layers). Set None as the global input to the internal models if they have no global inputs.')
            else:
                for l in range(self.L):
                    if l==0:
                        z=[[element] for element in np.array_split(x[l],chunk_num)]
                    else:
                        z_l=x[l]
                        z_m=[[]]*chunk_num
                        for m in range(len(z_l)):
                            if z_l[m]==None:
                                z_m=[i+j for i, j in zip(z_m,[[None]]*chunk_num)]
                            else:
                                z_m=[i+[j] for i, j in zip(z_m,np.array_split(z_l[m],chunk_num))]
                        z=[i+[j] for i,j in zip(z,z_m)]
        elif not isinstance(x, list):
            z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            res = pool.map(f, [[x, method, full_layer, sample_size] for x in z])
        if method == 'mean_var':
            if full_layer:
                combined_res=[]
                for comp in zip(*res):
                    combined_comp=[]
                    for layer in zip(*comp):
                        combined_comp.append(list(np.concatenate(worker) for worker in zip(*list(layer))))
                    combined_res.append(combined_comp)
                return tuple(combined_res)
            else:
                combined_res=[]
                for comp in zip(*res):
                    combined_res.append(list(np.concatenate(workers) for workers in zip(*list(comp))))
                return tuple(combined_res)
        elif method == 'sampling':
            if full_layer:
                combined_res=[]
                for layer in zip(*res):
                    combined_res.append(list(np.concatenate(workers,axis=1) for workers in zip(*list(layer))))
                return combined_res
            else:
                return list(np.concatenate(worker,axis=1) for worker in zip(*res))

    def predict(self,x,method='mean_var',full_layer=False,sample_size=50):
        """Implement predictions from the linked (D)GP model.

        Args:
            x (ndarray_or_list): a numpy 2d-array or a list.
            
                1. If **x** is a 2d-array, it is the global testing input set to the computer models in the first layer where 
                   each rows are input testing data points and columns are input dimensions across all computer models in 
                   the first layer of the system. In this case, it is assumed that **x** is the only global input to the computer 
                   system, i.e., there are no external global input to computer models in layers other than the first layer.
                2. If **x** is a list, it has *L* (the number of layers of a systems of computer models) elements. The first element
                   is a numpy 2d-array that represents the global testing input set to the computer models in the first layer. 
                   The remaining *L-1* elements are *L-1* sub-lists, each of which contains a number (same as the number of computer
                   models in the corresponding layer) of numpy 2d-arrays (rows being testing points and columns being input 
                   dimensions) that represent the external global testing input to the computer models in the corresponding layer. 
                   The order of 2d-arrays in each sub-list must be the same order of the emulators placed in the corresponding layer
                   of **all_layer** argument to :class:`.lgp` class. If there is no external global input to a certain computer model of the 
                   system, set `None` in the corresponding sub-list (i.e., layer) of **x**.   
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            full_layer (bool, optional): whether to output the predictions from all GP/DGP emulators in the system. Defaults to `False`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                Defaults to `50`.
            
        Returns:
            tuple_or_list: 
                if the argument **method** = '`mean_var`', a tuple is returned:
                    1. If **full_layer** = `False`, the tuple contains two lists, one for the predictive means 
                       and another for the predictive variances. Each list contains a number (same number of computer models in the
                       final layer of the system) of numpy 2d-arrays. Each 2d-array has its rows corresponding to global testing 
                       positions and columns corresponding to GP/DGP (or DGP+likelihood) output dimensions of the associated computer
                       model in the final layer;
                    2. If **full_layer** = `True`, the tuple contains two lists, one for the predictive means 
                       and another for the predictive variances. Each list contains *L* (i.e., the number of layers of the emulated system) 
                       sub-lists. Each sub-list represents a layer and contains a number (same number of computer models in the corresponding 
                       layer of the system) of numpy 2d-arrays. Each array has its rows corresponding to global testing positions and columns 
                       corresponding to GP/DGP (or DGP+likelihood in case of the final layer) output dimensions of the associated computer
                       model in the corresponding layer.
                if the argument **method** = '`sampling`', a list is returned:
                    1. If **full_layer** = `False`, the list contains a number (same number of computer models in the final layer of the system) of numpy 
                       3d-arrays. Each array corresponds to a computer model in the final layer, and has its 0-axis corresponding to the output 
                       dimensions of the GP/DGP (or DGP+likelihood) emulators, 1-axis corresponding to global testing positions, and 2-axis 
                       corresponding to samples of size **N** * **sample_size**;
                    2. If **full_layer** = `True`, the list contains *L* (i.e., the number of layers of the emulated system) sub-lists. Each sub-list 
                       represents a layer and contains a number (same number of computer models in the corresponding layer of the system) of 
                       numpy 3d-arrays. Each array corresponds to a computer model in the associated layer, and has its 0-axis corresponding 
                       to the output dimensions of the GP/DGP (or DGP+likelihood in case of the final layer) emulators, 1-axis corresponding 
                       to global testing positions, and 2-axis corresponding to samples of size **N** * **sample_size**.

        """
        if isinstance(x, list) and len(x)!=self.L:
            raise Exception('When test input is given as a list, it must contain global inputs to the all layers (even with no global inputs to internal layers). Set None as the global input to the internal models if they have no global inputs.')
        elif not isinstance(x, list):
            if x.ndim==1:
                raise Exception('The testing input has to be a numpy 2d-array.')
            x=[x]
            for num in self.num_model:
                x.append([None]*num) 
        if method=='mean_var':
            sample_size=1
        mean_pred=[]
        variance_pred=[]
        if method=='sampling':
            sample_pred=[]
        for s in range(len(self.all_layer_set)):
            one_imputed_all_layer=self.all_layer_set[s]
            if full_layer:
                if method=='mean_var':
                    mean_pred_oneN=[]
                    variance_pred_oneN=[]
                elif method=='sampling':
                    sample_pred_oneN=[]
            for l in range(self.L):
                layer=one_imputed_all_layer[l]
                n_model=len(layer)
                if l==self.L-1:
                    m_last_layer,v_last_layer=[], []
                    if method=='sampling':
                        sample_last_layer=[]
                else:
                    m_l,v_l=[], []
                    if method=='sampling':
                        sample_l=[]
                if l==0:
                    for k in range(n_model):
                        model=layer[k]
                        input_lk=x[l][:,model.local_input_idx]
                        if model.type=='gp':
                            m_lk, v_lk = self.gp_pred(input_lk,None,None,None,model.structure,self.nb_parallel)
                        elif model.type=='dgp':
                            _, _, m_lk, v_lk = self.dgp_pred(input_lk,None,None,None,model.structure,self.nb_parallel)
                        m_l.append(m_lk)
                        v_l.append(v_lk)
                        if method=='sampling' and full_layer:
                            row_num, col_num = np.shape(m_lk)
                            sample_lk=np.random.normal(m_lk,np.sqrt(v_lk),size=(sample_size, row_num, col_num))
                            sample_l.append(sample_lk.transpose(2,1,0))
                    if full_layer:
                        if method=='mean_var':
                            mean_pred_oneN.append(m_l)
                            variance_pred_oneN.append(v_l)
                        elif method=='sampling':
                            sample_pred_oneN.append(sample_l)
                    m_l_next, v_l_next=np.concatenate(m_l,axis=1), np.concatenate(v_l,axis=1)
                elif l==self.L-1:
                    for k in range(n_model):
                        model=layer[k]
                        external_input_lk=x[l][k]
                        m_input_lk, v_input_lk = m_l_next[:,model.local_input_idx], v_l_next[:,model.local_input_idx]
                        if model.type=='gp':
                            m_lk, v_lk=self.gp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,self.nb_parallel)
                            if method=='sampling':
                                row_num, col_num = np.shape(m_lk)
                                sample_lk=np.random.normal(m_lk,np.sqrt(v_lk),size=(sample_size, row_num, col_num)).transpose(2,1,0)
                        elif model.type=='dgp':
                            m_one_before_lk, v_one_before_lk, m_lk, v_lk = self.dgp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,self.nb_parallel)
                            if method=='sampling':
                                row_num, col_num = np.shape(m_lk)
                                sample_lk=np.empty((col_num, row_num, sample_size))
                                for count, kernel in enumerate(model.structure[-1]):
                                    if kernel.type=='gp':
                                        sample_lk[count,]=np.random.normal(m_lk[:,[count]],np.sqrt(v_one_before_lk[:,[count]]),size=(row_num,sample_size))
                                    elif kernel.type=='likelihood':
                                        dgp_sample=np.random.normal(m_one_before_lk,np.sqrt(v_one_before_lk),size=(sample_size, m_one_before_lk.shape[0], m_one_before_lk.shape[1]))
                                        sample_lk[count,]=np.array([kernel.sampling(dgp_sample[:,:,[kernel.input_dim]][i]) for i in range(sample_size)]).T
                        if method=='mean_var':                     
                            m_last_layer.append(m_lk)
                            v_last_layer.append(v_lk)
                        elif method=='sampling':
                            sample_last_layer.append(sample_lk)
                    if full_layer:
                        if method=='mean_var':
                            mean_pred_oneN.append(m_last_layer)
                            variance_pred_oneN.append(v_last_layer)
                        elif method=='sampling':
                            sample_pred_oneN.append(sample_last_layer)
                else:
                    for k in range(n_model):
                        model=layer[k]
                        external_input_lk=x[l][k]
                        m_input_lk, v_input_lk = m_l_next[:,model.local_input_idx], v_l_next[:,model.local_input_idx]
                        if model.type=='gp':
                            m_lk, v_lk = self.gp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,self.nb_parallel)
                        elif model.type=='dgp':
                            _, _, m_lk, v_lk = self.dgp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,self.nb_parallel)
                        m_l.append(m_lk)
                        v_l.append(v_lk)
                        if method=='sampling' and full_layer:
                            row_num, col_num = np.shape(m_lk)
                            sample_lk=np.random.normal(m_lk,np.sqrt(v_lk),size=(sample_size, row_num, col_num))
                            sample_l.append(sample_lk.transpose(2,1,0))
                    if full_layer:
                        if method=='mean_var':
                            mean_pred_oneN.append(m_l)
                            variance_pred_oneN.append(v_l)
                        elif method=='sampling':
                            sample_pred_oneN.append(sample_l)
                    m_l_next, v_l_next=np.concatenate(m_l,axis=1), np.concatenate(v_l,axis=1)
            if full_layer:
                if method=='mean_var':
                    mean_pred.append(mean_pred_oneN)
                    variance_pred.append(variance_pred_oneN)
                elif method=='sampling':
                    sample_pred.append(sample_pred_oneN)
            else:
                if method=='mean_var':
                    mean_pred.append(m_last_layer)
                    variance_pred.append(v_last_layer)
                elif method=='sampling':
                    sample_pred.append(sample_last_layer)
        if method=='mean_var':
            if full_layer:
                mu=[[np.mean(i,axis=0) for i in zip(*case_m)] for case_m in zip(*mean_pred)]
                sigma2=[[np.mean(np.square(i)+j,axis=0)-np.mean(i,axis=0)**2 for i,j in zip(zip(*case_m),zip(*case_v))] for case_m, case_v in zip(zip(*mean_pred),zip(*variance_pred))]
            else:
                mu=[np.mean(i,axis=0) for i in zip(*mean_pred)]
                sigma2=[np.mean(np.square(i)+j,axis=0)-np.mean(i,axis=0)**2 for i, j in zip(zip(*mean_pred),zip(*variance_pred))]
            return mu, sigma2
        elif method=='sampling':
            if full_layer:
                samples=[[np.concatenate(i,axis=2) for i in zip(*case_s)] for case_s in zip(*sample_pred)]
            else:
                samples=[np.concatenate(i,axis=2) for i in zip(*sample_pred)]
            return samples

    @staticmethod
    def gp_pred(x,m,v,z,structure,nb_parallel):
        """Compute predictive mean and variance from a GP emulator when the testing input is either deterministic or normally distributed.
        """
        if x is None:
            m,v=structure.linkgp_prediction(m=m,v=v,z=z,nb_parallel=nb_parallel)
        else:
            m,v=structure.gp_prediction(x=x,z=z)
        return m.reshape(-1,1),v.reshape(-1,1)
    
    @staticmethod
    def dgp_pred(x,m,v,z,structure,nb_parallel):
        """Compute predictive mean and variance from a DGP (DGP+likelihood) emulator when the testing input is either deterministic or normally distributed.
        """
        if x is None:
            M=len(m)
        else:
            M=len(x)
        L=len(structure)
        for l in range(L):
            layer=structure[l]
            n_kerenl=len(layer)
            if l==L-1:
                likelihood_gp_mean=np.empty((M,n_kerenl))
                likelihood_gp_var=np.empty((M,n_kerenl))
            else:
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
            if l==0:
                for k in range(n_kerenl):
                    kernel=layer[k]
                    if x is None:
                        m_k,v_k=kernel.linkgp_prediction(m=m,v=v,z=z,nb_parallel=nb_parallel)
                    else:
                        m_k,v_k=kernel.gp_prediction(x=x,z=z)
                    overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            elif l==L-1:
                for k in range(n_kerenl):
                    kernel=layer[k]
                    m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                    if kernel.type=='gp':
                        if kernel.connect is not None:
                            if x is None:
                                D=np.shape(m)[1]
                                idx1,idx2=kernel.connect[kernel.connect<=(D-1)],kernel.connect[kernel.connect>(D-1)]
                                if idx1.size==0:
                                    m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,idx2-D],nb_parallel=nb_parallel)
                                elif idx2.size==0:
                                    m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=None,nb_parallel=nb_parallel)
                                else:
                                    m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=z[:,idx2-D],nb_parallel=nb_parallel)
                            else:
                                m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=x[:,kernel.connect],nb_parallel=nb_parallel)
                        else:
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=None,nb_parallel=nb_parallel)
                        likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                    elif kernel.type=='likelihood':
                        m_k,v_k=kernel.prediction(m=m_k_in,v=v_k_in)
                        likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
            else:
                for k in range(n_kerenl):
                    kernel=layer[k]
                    m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                    if kernel.connect is not None:
                        if x is None:
                            D=np.shape(m)[1]
                            idx1,idx2=kernel.connect[kernel.connect<=(D-1)],kernel.connect[kernel.connect>(D-1)]
                            if idx1.size==0:
                                m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,idx2-D],nb_parallel=nb_parallel)
                            elif idx2.size==0:
                                m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=None,nb_parallel=nb_parallel)
                            else:
                                m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=z[:,idx2-D],nb_parallel=nb_parallel)
                        else:
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=x[:,kernel.connect],nb_parallel=nb_parallel)
                    else:
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=None,nb_parallel=nb_parallel)
                    overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
        return overall_test_input_mean, overall_test_input_var, likelihood_gp_mean, likelihood_gp_var

         