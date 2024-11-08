import multiprocess.context as ctx
import platform
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil
from .imputation import imputer
import copy
from numba import set_num_threads
from .utils import have_same_shape
from contextlib import contextmanager

class container:
    """
    Class to contain the trained GP or DGP emulator of a computer model for linked (D)GP emulation.

    Args:
        structure (list): a list that contains the trained structure of GP or DGP of a computer model. For GP, 
            this is the list exported from the :meth:`.export` method of the :class:`.gp` class. For DGP, this is the list exported 
            from the :meth:`.estimate` of the :class:`.dgp` class.
        local_input_idx (ndarray_or_list): a numpy 1d-array or a list:

            1. If **local_input_idx** is a 1d-array, it specifies the indices of outputs (a 2d-array) 
               produced by all emulators in the feeding layer that are input to the emulator
               represented by the **structure** argument. The indices should be ordered in such a way that the extracted
               output from the feeding layer is sorted in the same order as the training input used for the GP/DGP 
               emulator that the **structure** argument represents. When the emulator is in the first
               layer, **local_input_idx** gives the indices of its input in the global testing input set, see :meth:`.lgp.predict`
               for descriptions of the global testing input set. 
            2. If **local_input_idx** is a list, the emulator must be in layer 2 or deeper layers. The list should have a number 
               (the same number of preceding layers, e.g., when an emulator is in the second layer, the list is of length 1) of elements. Each
               element is a 1d-array that specifies the indices of outputs produced by all emulators in the corresponding layer 
               that feed to the emulator represented by the **structure** argument. If there is no output connections from a certain layer,
               set `None` instead in the list. 
               
            Defaults to `None`. When the argument is `None`, one needs to set its value using the :meth:`.set_local_input`. 
        block (bool, optional): whether to use the blocked (layer-wise) ESS for the imputations. Defaults to `True`.
    """
    def __init__(self, structure, local_input_idx=None, block=True):
        if len(structure)==1:
            self.type='gp'
            self.structure=structure[0]
            if self.structure.vecch:
                self.vecch=True
            else:
                self.vecch=False
        else:
            self.type='dgp'
            self.structure=structure
            if self.structure[0][0].vecch:
                self.vecch=True
            else:
                self.vecch=False
            self.imp=imputer(self.structure, block)
            if self.vecch:
                (self.imp).update_ord_nn()
            self.imp.sample(burnin=50)
        self.local_input_idx=local_input_idx

    def __setstate__(self, state):
        if 'vecch' not in state:
            state['vecch'] = False
        self.__dict__.update(state)

    def to_vecchia(self):
        """Convert the container to the Vecchia mode.
        """
        if not self.vecch:
            self.vecch=True
            if self.type == 'gp':
                self.structure.vecch = self.vecch
            elif self.type == 'dgp':
                for layer in self.structure:
                    for kernel in layer:
                        if kernel.type == 'gp':
                            kernel.vecch = self.vecch

    def remove_vecchia(self):
        """Remove the Vecchia mode from the container.
        """
        if self.vecch:
            self.vecch = False
            if self.type == 'gp':
                self.structure.vecch = self.vecch
                self.structure.compute_stats()
            elif self.type == 'dgp':
                for layer in self.structure:
                    for kernel in layer:
                        if kernel.type == 'gp':
                            kernel.vecch = self.vecch

    def set_local_input(self, idx, new = False):
        """Set the **local_input_idx** argument and optionally output a copy of the container with a different **local_input_idx**.

        Args:
            idx (ndarray_or_list): see :class:`.container` for details.
            new (bool, optional): whether to output a copy of the container with a different **local_input_idx**. Defaults to `False`.

        Remark: 
           This method is useful in the following scenarios:
           
                1. when different models are emulated by different teams. Each team can create the container
                   of their model even without knowing how different models are connected together. When this information is available and
                   containers of different emulators are collected, the connections between emulators can then be set by assigning
                   values to **local_input_idx** of each container with this method.
                2. when **local_input_idx** was not correctly specified when the container was created, one can correct **local_input_idx** 
                   swiftly without recreating it.
                3. when the same emulator in the container is repeatedly used in a system, one can set **new** to `True` to create copies of the
                   container by assigning different **local_input_idx** to the copies swiftly without generating the containers repeatedly.
        """
        if new:
            container_cp = copy.copy(self)
            container_cp.local_input_idx=idx
            return(container_cp)
        else:
            self.local_input_idx=idx

    def __copy__(self):
        new_inst = type(self).__new__(self.__class__)
        new_inst.type = self.type
        new_inst.structure = self.structure
        new_inst.vecch = self.vecch
        if self.type=='dgp':
            new_inst.imp = self.imp
        new_inst.local_input_idx = copy.copy(self.local_input_idx)
        return new_inst

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
            Defaults to `10`.
    """
    def __init__(self, all_layer, N=10):
        self.L=len(all_layer)
        self.all_layer=all_layer
        self.num_model=[]
        for l in range(1,self.L):
            self.num_model.append(len(all_layer[l]))
        if np.sum(np.concatenate([[cont.type=='dgp' for cont in all_layer[l]] for l in range(self.L)]))==0:
            N=1
        self.all_layer_set=[]
        with self.temp_all_layer() as temp_all_layer:
            for _ in range(N):
                one_imputation=[]
                for l in range(self.L):
                    layer=[]
                    for cont in temp_all_layer[l]:
                        if cont.type=='gp':
                            layer.append(copy.deepcopy(cont))
                        elif cont.type=='dgp':
                            if cont.vecch:
                                (cont.imp).update_ord_nn()
                            (cont.imp).sample()
                            if not cont.vecch:
                                (cont.imp).key_stats()
                            layer.append(copy.deepcopy(cont))
                    one_imputation.append(layer)
                self.all_layer_set.append(one_imputation)

    def __setstate__(self, state):
        if 'nb_parallel' in state:
            del state['nb_parallel']
        self.__dict__.update(state)

    @contextmanager
    def temp_all_layer(self):
        original_state = copy.deepcopy(self.all_layer)
        try:
            yield original_state
        finally:
            pass

    def set_vecchia(self, mode):
        """Convert the (D)GP emulators in the linked system to Vecchia or non-Vecchia mode.

        Args:
            mode (bool_or_list): a bool or a list of bools.

            1. If **mode** is a bool, it indicates whether to set all (D)GP emulators in the linked system to 
               the Vecchia (`True`) or non-Vecchia (`False`) mode.
            2. If **mode** is a list, it is a list contains *L* (the number of layers of a systems of computer models) sub-lists, 
               each of which represents a layer and contains same number of bools as that of the GP/DGP emulators of computer models 
               in the same layer. The list has the same shape as the **all_layer** argument of :class:`.lgp` class.

        """
        if isinstance(mode, list):
            if not have_same_shape(self.all_layer, mode):
                raise Exception('mode has a different shape as all_layer.')
        else:
            mode = [[mode for _ in layer] for layer in self.all_layer]
        for layer, mode_layer in zip(self.all_layer, mode):
            for cont, cont_mode in zip(layer, mode_layer):
                if cont_mode:
                    cont.to_vecchia()
                else:
                    cont.remove_vecchia()
        for one_imputed in self.all_layer_set:
            for layer, mode_layer in zip(one_imputed, mode):
                for cont, cont_mode in zip(layer, mode_layer):
                    if cont_mode:
                        cont.to_vecchia()
                    else:
                        cont.remove_vecchia()
                        if cont.type=='dgp':
                            (cont.imp).key_stats()
    
    def ppredict(self,x,method='mean_var',full_layer=False,sample_size=50,m=50,chunk_num=None,core_num=None):
        """Implement parallel predictions from the trained DGP model.

        Args:
            x, method, full_layer, sample_size, m: see descriptions of the method :meth:`.lgp.predict`.
            chunk_num (int, optional): the number of chunks that the testing input array **x** will be divided into. 
                Defaults to `None`. If not specified, the number of chunks is set to **core_num**. 
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.

        Returns:
            Same as the method `predict`.
        """
        os_type = platform.system()
        if os_type in ['Darwin', 'Linux']:
            ctx._force_start_method('forkserver')
        total_cores = psutil.cpu_count(logical = False)
        if core_num==None:
            core_num=total_cores//2
        if chunk_num==None:
            chunk_num=core_num
        if chunk_num<core_num:
            core_num=chunk_num
        num_thread = total_cores // core_num
        def f(params):
            x, method, full_layer, sample_size, m = params
            set_num_threads(num_thread)
            return self.predict(x, method, full_layer, sample_size, m)
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
                            if z_l[m] is None:
                                z_m=[i+j for i, j in zip(z_m,[[None]]*chunk_num)]
                            else:
                                z_m=[i+[j] for i, j in zip(z_m,np.array_split(z_l[m],chunk_num))]
                        z=[i+[j] for i,j in zip(z,z_m)]
        elif not isinstance(x, list):
            z=np.array_split(x,chunk_num)
        with Pool(core_num) as pool:
            res = pool.map(f, [[x, method, full_layer, sample_size, m] for x in z])
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

    def predict(self,x,method='mean_var',full_layer=False,sample_size=50,m=50):
        """Implement predictions from the linked (D)GP model.

        Args:
            x (ndarray_or_list): a numpy 2d-array or a list.
            
                1. If **x** is a 2d-array, it is the global testing input set to the computer emulators in the first layer where 
                   each rows are input testing data points and columns are input dimensions across all computer emulators in 
                   the first layer of the system. In this case, it is assumed that **x** is the only global input to the computer 
                   system, i.e., there are no external global input to computer emulators in layers other than the first layer.
                2. If **x** is a list, it has *L* (the number of layers of a systems of emulators) elements. The first element
                   is a numpy 2d-array that represents the global testing input set to the computer emulators in the first layer. 
                   The remaining *L-1* elements are *L-1* sub-lists, each of which contains a number (same as the number of computer
                   emulators in the corresponding layer) of numpy 2d-arrays (rows being testing points and columns being input 
                   dimensions) that represent the external global testing input to the computer models in the corresponding layer. 
                   The order of 2d-arrays in each sub-list must be the same order of the emulators placed in the corresponding layer
                   of **all_layer** argument to :class:`.lgp` class. If there is no external global input to a certain computer emulator of the 
                   system, set `None` in the corresponding sub-list (i.e., layer) of **x**.   
            method (str, optional): the prediction approach: mean-variance (`mean_var`) or sampling 
                (`sampling`) approach. Defaults to `mean_var`.
            full_layer (bool, optional): whether to output the predictions from all GP/DGP emulators in the system. Defaults to `False`.
            sample_size (int, optional): the number of samples to draw for each given imputation if **method** = '`sampling`'.
                Defaults to `50`.
            m (int, optional): the size of the conditioning set for predictions if the DGP was built under the Vecchia approximation. Defaults to `50`.
            
        Returns:
            tuple_or_list: 
                if the argument **method** = '`mean_var`', a tuple is returned:

                    1. If **full_layer** = `False`, the tuple contains two lists, one for the predictive means 
                       and another for the predictive variances. Each list contains a number (same number of emulators in the
                       final layer of the system) of numpy 2d-arrays. Each 2d-array has its rows corresponding to global testing 
                       positions and columns corresponding to output dimensions of the associated emulator in the final layer;
                    2. If **full_layer** = `True`, the tuple contains two lists, one for the predictive means 
                       and another for the predictive variances. Each list contains *L* (i.e., the number of layers of the emulated system) 
                       sub-lists. Each sub-list represents a layer and contains a number (same number of emulators in the corresponding 
                       layer of the system) of numpy 2d-arrays. Each array has its rows corresponding to global testing positions and columns 
                       corresponding to output dimensions of the associated GP/DGP emulator in the corresponding layer.
                if the argument **method** = '`sampling`', a list is returned:
                    
                    1. If **full_layer** = `False`, the list contains a number (same number of emulators in the final layer of the system) of numpy 
                       3d-arrays. Each array corresponds to an emulator in the final layer, and has its 0-axis corresponding to the output 
                       dimensions of the GP/DGP emulator, 1-axis corresponding to global testing positions, and 2-axis 
                       corresponding to samples of size **N** * **sample_size**;
                    2. If **full_layer** = `True`, the list contains *L* (i.e., the number of layers of the emulated system) sub-lists. Each sub-list 
                       represents a layer and contains a number (same number of emulators in the corresponding layer of the system) of 
                       numpy 3d-arrays. Each array corresponds to an emulator in the associated layer, and has its 0-axis corresponding 
                       to the output dimensions of the GP/DGP emulators, 1-axis corresponding to global testing positions, and 2-axis 
                       corresponding to samples of size **N** * **sample_size**.

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
                        if isinstance(model.local_input_idx, list):
                            raise Exception('When an emulator is in the first layer, local_input_idx must be a 1d-array.') 
                        input_lk=x[l][:,model.local_input_idx]
                        if model.type=='gp':
                            m_lk, v_lk = self.gp_pred(input_lk,None,None,None,model.structure,m)
                        elif model.type=='dgp':
                            _, _, m_lk, v_lk = self.dgp_pred(input_lk,None,None,None,model.structure,m)
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
                    m_l_next, v_l_next=[np.concatenate(m_l,axis=1)], [np.concatenate(v_l,axis=1)]
                elif l==self.L-1:
                    for k in range(n_model):
                        model=layer[k]
                        if isinstance(model.local_input_idx, list):
                            if len(model.local_input_idx)!=self.L-1:
                                raise Exception('local_input_idx should be a list that has length of %i.' % (self.L-1))
                            else:
                                local_input_idx = model.local_input_idx
                        else:
                            local_input_idx=[None]*(self.L-2)
                            local_input_idx.append(model.local_input_idx)
                        external_input_lk=x[l][k]
                        m_input_lk, v_input_lk = [], []
                        for i in range(l):
                            idx = local_input_idx[i]
                            if idx is not None:
                                m_input_lk.append( m_l_next[i][:,idx] )
                                v_input_lk.append( v_l_next[i][:,idx] )
                        m_input_lk, v_input_lk = np.concatenate(m_input_lk, axis=1), np.concatenate(v_input_lk, axis=1)
                        #m_input_lk, v_input_lk = None if not m_input_lk else np.concatenate(m_input_lk, axis=1), None if not v_input_lk else np.concatenate(v_input_lk, axis=1)
                        if model.type=='gp':
                            m_lk, v_lk=self.gp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,m)
                            if method=='sampling':
                                row_num, col_num = np.shape(m_lk)
                                sample_lk=np.random.normal(m_lk,np.sqrt(v_lk),size=(sample_size, row_num, col_num)).transpose(2,1,0)
                        elif model.type=='dgp':
                            m_one_before_lk, v_one_before_lk, m_lk, v_lk = self.dgp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,m)
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
                        if isinstance(model.local_input_idx, list):
                            if len(model.local_input_idx)!=l:
                                raise Exception('local_input_idx should be a list that has length of %i.' % (l))
                            else:
                                local_input_idx = model.local_input_idx
                        else:
                            local_input_idx=[None]*(l-1)
                            local_input_idx.append(model.local_input_idx)
                        external_input_lk=x[l][k]
                        m_input_lk, v_input_lk = [], []
                        for i in range(l):
                            idx = local_input_idx[i]
                            if idx is not None:
                                m_input_lk.append( m_l_next[i][:,idx] )
                                v_input_lk.append( v_l_next[i][:,idx] )
                        m_input_lk, v_input_lk = np.concatenate(m_input_lk, axis=1), np.concatenate(v_input_lk, axis=1)
                        if model.type=='gp':
                            m_lk, v_lk = self.gp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,m)
                        elif model.type=='dgp':
                            _, _, m_lk, v_lk = self.dgp_pred(None,m_input_lk,v_input_lk,external_input_lk,model.structure,m)
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
                    m_l_next.append(np.concatenate(m_l,axis=1))
                    v_l_next.append(np.concatenate(v_l,axis=1))
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
    def gp_pred(x,m,v,z,structure,m_pred):
        """Compute predictive mean and variance from a GP emulator when the testing input is either deterministic or normally distributed.
        """
        structure.pred_m = m_pred
        if x is None:
        #    if m is None:
        #        m,v=structure.gp_prediction(x=z,z=None)
        #    else:
            m,v=structure.linkgp_prediction(m=m,v=v,z=z)
        else:
            m,v=structure.gp_prediction(x=x,z=z)
        return m.reshape(-1,1),v.reshape(-1,1)
    
    @staticmethod
    def dgp_pred(x,m,v,z,structure,pred_m):
        """Compute predictive mean and variance from a DGP (DGP+likelihood) emulator when the testing input is either deterministic or normally distributed.
        """
        if x is None:
        #    if m is None:
        #        M = len(z)
        #    else:
            M=len(m)
        else:
            M=len(x)
        L=len(structure)
        internal_idx=structure[0][0].input_dim
        external_idx=structure[0][0].connect
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
                    kernel.pred_m = pred_m
                    if x is None:
                    #    if m is None:
                    #        m_k,v_k=kernel.gp_prediction(x=z,z=None)
                    #    else:
                        m_k,v_k=kernel.linkgp_prediction(m=m,v=v,z=z)
                    else:
                        m_k,v_k=kernel.gp_prediction(x=x,z=z)
                    overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            elif l==L-1:
                for k in range(n_kerenl):
                    kernel=layer[k]
                    m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                    if kernel.type=='gp':
                        kernel.pred_m = pred_m
                        if kernel.connect is not None:
                            if x is None:
                            #    if m is None:
                            #        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,kernel.connect])
                            #    else:
                                if external_idx is None:
                                    idx=np.where(kernel.connect[:, None] == internal_idx[None, :])[1]
                                    m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx],v_z=v[:,idx],z=None)
                                else:
                                    idx1 = np.where(kernel.connect[:, None] == internal_idx[None, :])[1]
                                    idx2 = np.where(kernel.connect[:, None] == external_idx[None, :])[1]
                                    if idx1.size==0:
                                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,idx2])
                                    elif idx2.size==0:
                                        m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=None)
                                    else:
                                        m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=z[:,idx2])
                            else:
                                m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=x[:,kernel.connect])
                        else:
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=None)
                        likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
                    elif kernel.type=='likelihood':
                        m_k,v_k=kernel.prediction(m=m_k_in,v=v_k_in)
                        likelihood_gp_mean[:,k],likelihood_gp_var[:,k]=m_k,v_k
            else:
                for k in range(n_kerenl):
                    kernel=layer[k]
                    kernel.pred_m = pred_m
                    m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                    if kernel.connect is not None:
                        if x is None:
                        #    if m is None:
                        #        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,kernel.connect])
                        #    else:
                            D=np.shape(m)[1]
                            idx1,idx2=kernel.connect[kernel.connect<=(D-1)],kernel.connect[kernel.connect>(D-1)]
                            if idx1.size==0:
                                m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z[:,idx2-D])
                            elif idx2.size==0:
                                m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=None)
                            else:
                                m_k,v_k=kernel.linkgp_prediction_full(m=m_k_in,v=v_k_in,m_z=m[:,idx1],v_z=v[:,idx1],z=z[:,idx2-D])
                        else:
                            m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=x[:,kernel.connect])
                    else:
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=None)
                    overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
        return overall_test_input_mean, overall_test_input_var, likelihood_gp_mean, likelihood_gp_var

         