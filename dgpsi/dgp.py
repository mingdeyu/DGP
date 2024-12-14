import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import copy
from .imputation import imputer
from .kernel_class import kernel as ker
from .kernel_class import combine
from .functions import cond_mean
from .utils import NystromKPCA
from .vecchia import cond_mean_vecch
from sklearn.decomposition import KernelPCA
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import cho_solve
import multiprocess.context as ctx
import platform
from pathos.multiprocessing import ProcessingPool as Pool
import psutil  
from numba import set_num_threads

class dgp:
    """
    Class that contains the deep GP hierarchy for stochastic imputation inference.

    Args:
        X (ndarray): a numpy 2d-array where each row is an input data point and 
            each column is an input dimension. 
        Y (ndarray): a numpy 2d-arrays containing observed output data. 
            The 2d-array has it rows being output data points and columns being output dimensions 
            (with the number of columns equals to the number of GP nodes in the final layer). 
        all_layer (list, optional): a list contains *L* (the number of layers) sub-lists, each of which contains 
            the GPs defined by the :class:`.kernel` class in that layer. The sub-lists are placed in the list 
            in the same order of the specified DGP model. The final layer of DGP hierarchy can be set to a likelihood
            layer by putting an object created by a likelihood class (in :mod:`likelihood_class`) into the final sub-list of **all_layer**. 
            Defaults to `None`. If a DGP structure is not provided, an input-connected two-layered DGP 
            structure (for deterministic model emulation without a likelihood layer), where the number of GP nodes in the first layer equals 
            to the dimension of **X**, is automatically constructed.
        check_rep (bool, optional): whether to check the repetitions in the dataset, i.e., if one input
            position has multiple outputs. Defaults to `True`.
        block (bool, optional): whether to use the blocked (layer-wise) ESS for the imputations during the training. Defaults to `True`.
        vecchia (bool): a bool indicating if Vecchia approximation will be used. Defaults to `False`. 
        m (int): an integer that gives the size of the conditioning set for the Vecchia approximation in the training. Defaults to `25`. 
        ord_fun (function, optional): a function that decides the ordering of the input of the GP nodes in the DGP structure for the Vecchia approximation.
            If set to `None`, then the default random ordering is used. Defaults to `None`.
    Remark:
        This class is used for DGP structures, in which internal I/O are unobservable. When some internal layers
        are fully observable, the DGP model reduces to linked (D)GP model. In such a case, use :class:`.lgp` class for 
        inference where one can have separate input/output training data for each (D)GP. See :class:`.lgp` class for 
        implementation details. 

    Examples:
        To build a list that represents a three-layer DGP with three GPs in the first two layers and
        one GP (i.e., only one dimensional output) in the final layer, do::

            from kernel_class import kernel, combine
            layer1, layer2, layer3=[],[],[]
            for _ in range(3):
                layer1.append(kernel(length=np.array([1])))
            for _ in range(3):
                layer2.append(kernel(length=np.array([1])))
            layer3.append(kernel(length=np.array([1])))
            all_layer=combine(layer1,layer2,layer3) 

    """

    def __init__(self, X, Y, all_layer=None, check_rep=True, block=True, vecchia=False, m=25, ord_fun=None):
        self.Y=Y
        if isinstance(self.Y, list):
            if len(self.Y)==1:
                self.Y=self.Y[0]
            else:
                raise Exception('Y has to be a numpy 2d-array rather than a list. The list version of Y (for linked emulation) has been reduced. Please use the dedicated lgp class for linked emulation.')
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        #if len(X)>=500 and rff==False:
        #    print('Your training data size is greater than %i, you might want to consider random Fourier features for a faster training by setting "rff=True".' % (500))
        #elif len(X)<=500 and rff:
        #    print('Your training data size is smaller than %i, you may not gain much on the computation by using random Fourier features at the expense of accuracy.' % (500))
        self.check_rep=check_rep
        self.indices=None
        if self.check_rep:
            X0, indices, counts = np.unique(X, return_inverse=True,return_counts=True,axis=0)
            if len(X0) != len(X):
                self.X = X0
                self.indices=indices
                self.counts=counts
                self.max_rep=np.max(counts)
            else:  
                self.X=X
        else:
            self.X=X
        self.vecch=vecchia
        self.n_data=self.X.shape[0]
        #if self.n_data>=1e4:
        #    self.nn_method = 'approx'
        #else:
        self.nn_method = 'exact'
        self.m=min(m, self.n_data-1)
        self.ord_fun=ord_fun
        if all_layer is None:
            D, Y_D=np.shape(self.X)[1], np.shape(self.Y)[1]
            layer1 = [ker(length=np.array([1.])) for _ in range(D)]
            layer2 = [ker(length=np.array([1.]),scale_est=True,connect=np.arange(D)) for _ in range(Y_D)]
            all_layer=combine(layer1,layer2)
        self.all_layer=all_layer
        self.n_layer=len(self.all_layer)
        if self.all_layer[-1][0].name == 'Categorical':
            self.all_layer[-1][0].class_encoder = LabelEncoder()
            self.Y = self.all_layer[-1][0].class_encoder.fit_transform(self.Y.flatten()).reshape(-1,1)
            if self.all_layer[-1][0].num_classes is None:
                self.all_layer[-1][0].num_classes = len(self.all_layer[-1][0].class_encoder.classes_)
        self.initialize()
        self.block=block
        self.imp=imputer(self.all_layer, self.block)
        (self.imp).sample(burnin=10)
        self.compute_r2()
        self.N=0
        self.burnin=None

    def __setstate__(self, state):
        if 'block' not in state:
            state['block'] = True
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
        if 'max_rep' not in state:
            state['max_rep'] = None
        if 'counts' not in state:
            state['counts'] = None
        if 'rff' in state:
            del state['rff']
        if 'M' in state:
            del state['M']
        self.__dict__.update(state)

    def initialize(self):
        """Initialise all_layer attribute for training.
        """
        global_in=self.X
        In=self.X
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            num_kernel=len(layer)
            if l!=self.n_layer-1:
                if l==self.n_layer-2 and num_kernel==2 and len(self.all_layer[l+1])==1 and self.all_layer[l+1][0].name=='Hetero':
                    Out = np.empty((np.shape(In)[0], num_kernel))
                    if self.indices is None:
                        Out[:,0] = self.Y.flatten()
                        Out[:,1] = np.log(np.var(self.Y.flatten()))
                    else:
                        sum_Y = np.zeros((np.shape(In)[0], 1))
                        sum_Y2 = np.zeros((np.shape(In)[0], 1))
                        # Accumulate the sum and sum of squares for each group
                        np.add.at(sum_Y, self.indices, self.Y)
                        np.add.at(sum_Y2, self.indices, self.Y**2)
                        # Compute the mean and variance for each group
                        mean_Y = sum_Y / self.counts[:, None]
                        mean_Y2 = sum_Y2 / self.counts[:, None]
                        variance_Y = mean_Y2 - mean_Y**2
                        variance_Y[variance_Y == 0] = np.var(self.Y)
                        Out[:,0] = mean_Y.flatten()
                        Out[:,1] = np.log(variance_Y.flatten())
                    if self.all_layer[l+1][0].input_dim is not None:
                        Out = Out[:,self.all_layer[l+1][0].input_dim]
                elif l==self.n_layer-2 and len(self.all_layer[l+1])==1 and self.all_layer[l+1][0].name=='Categorical':
                    if self.all_layer[l+1][0].num_classes==2:
                        if num_kernel != 1:
                            raise Exception('You need one GP node to feed the categorical likelihood node.')
                    else:
                        if num_kernel != self.all_layer[l+1][0].num_classes:
                            raise Exception('You need ' + str(self.all_layer[l+1][0].num_classes) + ' GP nodes to feed the ' + kernel.name + ' likelihood node.')
                    if self.indices is None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            lgm = LogisticRegression().fit(self.X, self.Y.flatten())
                        w, b = lgm.coef_, lgm.intercept_
                        Out = np.dot(self.X, w.T) + b
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            lgm = LogisticRegression().fit(self.X[self.indices,:], self.Y.flatten())
                        w, b = lgm.coef_, lgm.intercept_
                        Out = np.dot(self.X, w.T) + b
                else:
                    if np.shape(In)[1]==num_kernel:
                        Out=copy.copy(In)
                    elif np.shape(In)[1]>num_kernel:
                        if self.vecch or self.n_data>=500:
                            pca=NystromKPCA(n_components=num_kernel)
                            Out=pca.fit_transform(In)
                        else:
                            pca=KernelPCA(n_components=num_kernel, kernel='sigmoid')
                            Out=pca.fit_transform(In)
                    else:
                        Out=np.concatenate((In, In[:,np.random.choice(np.shape(In)[1],num_kernel-np.shape(In)[1])]),1)
            for k in range(num_kernel):
                kernel=layer[k]
                if l==self.n_layer-1 and self.indices is not None:
                    kernel.rep=self.indices
                    #kernel.rep_sp=rep_sp(kernel.rep)
                if kernel.input_dim is not None:
                    if l==self.n_layer-1:
                        if kernel.type=='likelihood':
                            if kernel.name=='Poisson' and len(kernel.input_dim)!=1:
                                raise Exception('You need one and only one GP node to feed the ' + kernel.name + ' likelihood node.')
                            elif (kernel.name=='Hetero' or kernel.name=='NegBin') and len(kernel.input_dim)!=2:
                                raise Exception('You need two and only two GP nodes to feed the ' + kernel.name + ' likelihood node.')
                        if kernel.rep is None:
                            kernel.input=In[:,kernel.input_dim]
                        else:
                            kernel.input=In[kernel.rep,:][:,kernel.input_dim]
                    else:
                        kernel.input=In[:,kernel.input_dim]
                else:
                    if l==self.n_layer-1:
                        kernel.input_dim=np.arange(np.shape(In)[1])
                        if kernel.type=='likelihood':
                            if kernel.name=='Poisson' and len(kernel.input_dim)!=1:
                                raise Exception('You need one and only one GP node to feed the ' + kernel.name + ' likelihood node.')
                            elif (kernel.name=='Hetero' or kernel.name=='NegBin') and len(kernel.input_dim)!=2:
                                raise Exception('You need two and only two GP nodes to feed the ' + kernel.name + ' likelihood node.')
                        if kernel.rep is None:
                            kernel.input=copy.copy(In)
                        else:
                            kernel.input=In[kernel.rep,:]
                    else:
                        kernel.input=copy.copy(In)
                        kernel.input_dim=np.arange(np.shape(In)[1])
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            if kernel.rep is None:
                                kernel.global_input=global_in[:,kernel.connect]
                            else:
                                kernel.global_input=global_in[kernel.rep,:][:,kernel.connect]
                        else:
                            if l==0 and len(np.intersect1d(kernel.connect,kernel.input_dim))!=0:
                                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
                            kernel.global_input=global_in[:,kernel.connect]
                    kernel.vecch, kernel.m, kernel.nn_method = self.vecch, self.m, self.nn_method
                    if self.ord_fun is not None:
                        kernel.ord_fun = self.ord_fun
                    kernel.D=np.shape(kernel.input)[1]
                    if kernel.connect is not None:
                        kernel.D+=len(kernel.connect)
                    if kernel.vecch:
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        kernel.max_rep = self.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                if l==self.n_layer-1:
                    kernel.output=self.Y[:,[k]]
                else:
                    kernel.output=Out[:,[k]]
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        p=np.shape(kernel.input)[1]
                        if kernel.global_input is not None:
                            p+=np.shape(kernel.global_input)[1]
                        b=1/len(kernel.output)**(1/p)*(kernel.prior_coef+p)
                        kernel.prior_coef=np.concatenate((kernel.prior_coef, b))
                        kernel.compute_cl()
                    #if kernel.scale_est:
                    #    kernel.compute_scale() 
                    kernel.para_path=np.atleast_2d(np.concatenate((kernel.scale,kernel.length,kernel.nugget)))
            if l!=self.n_layer-1:
                In=copy.copy(Out)

    def to_vecchia(self, m=25, ord_fun=None):
        """Convert the DGP structure to the Vecchia mode.

        Args:
            m (int): an integer that gives the size of the conditioning set for the Vecchia approximation in the training. Defaults to `25`. 
            ord_fun (function, optional): a function that decides the ordering of the input of the GP nodes in the DGP structure for the Vecchia approximation. If set to `None`, then the default random ordering is used. Defaults to `None`.
        """
        if self.vecch:
            raise Exception('The DGP structure is already in Vecchia mode.')
        else:
            self.vecch = True
            self.m = min(m, self.n_data-1)
            self.ord_fun = ord_fun
            if self.indices is not None:
                self.max_rep = np.max(np.bincount(self.indices))
            n_layer = len(self.all_layer)
            for l in range(n_layer):
                layer = self.all_layer[l]
                for k, kernel in enumerate(layer):
                    if kernel.type == 'gp':
                        kernel.vecch, kernel.m = self.vecch, self.m
                        kernel.ord_fun = self.ord_fun
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        kernel.max_rep = self.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)

    def remove_vecchia(self):
        """Remove the Vecchia mode from the DGP structure.
        """
        if self.vecch:
            self.vecch = False
            for layer in self.all_layer:
                for kernel in layer:
                    if kernel.type == 'gp':
                        kernel.vecch = self.vecch
        else:
            raise Exception('The DGP structure is already in non-Vecchia mode.')

    def update_all_layer(self, all_layer):
        """Update the class with a new dgp structure with given hyperparameter and latent layer values.
        """
        self.all_layer=all_layer
        self.n_layer=len(self.all_layer)
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            for k, kernel in enumerate(layer):
                if l==self.n_layer-1 and kernel.rep is not None:
                    self.indices=kernel.rep
                if kernel.type=='gp':
                    kernel.para_path=np.atleast_2d(np.concatenate((kernel.scale,kernel.length,kernel.nugget)))
                    kernel.D=np.shape(kernel.input)[1]
                    if kernel.connect is not None:
                        kernel.D+=len(kernel.connect)
                    if kernel.vecch:
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        if kernel.max_rep is None:
                                            self.max_rep = np.max(np.bincount(self.indices))
                                            kernel.max_rep = self.max_rep
                                        else:
                                            self.max_rep = kernel.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                    if kernel.prior_name=='ref':
                        p=np.shape(kernel.input)[1]
                        if kernel.global_input is not None:
                            p+=np.shape(kernel.global_input)[1]
                        kernel.prior_coef[1]=1/len(kernel.output)**(1/p)*(kernel.prior_coef[0]+p)
                        kernel.compute_cl()
        self.imp=imputer(self.all_layer, self.block)
        (self.imp).sample(burnin=10)
        self.compute_r2()
        self.N=0
        self.burnin=None

    def update_xy(self, X, Y, reset=False):
        """Update the trained DGP with new input and output data.

        Args:
            X (ndarray): a numpy 2d-array where each row is an input data point and 
                each column is an input dimension. 
            Y (ndarray): a numpy 2d-arrays containing observed output data. 
                The 2d-array has it rows being output data points and columns being output dimensions 
                (with the number of columns equals to the number of GP nodes in the final layer). 
            reset (bool, optional): whether to reset latent layers and hyperparameter values of the DGP emulator. Defaults to `False`. 
        """
        self.Y=Y
        if isinstance(self.Y, list):
            if len(self.Y)==1:
                self.Y=self.Y[0]
            else:
                raise Exception('Y has to be a numpy 2d-array rather than a list. The list version of Y (for linked emulation) has been reduced. Please use the dedicated lgp class for linked emulation.')
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        if self.all_layer[-1][0].name == 'Categorical':
            self.Y = self.all_layer[-1][0].class_encoder.transform(self.Y.flatten()).reshape(-1,1)
        self.indices=None
        origin_X=(self.X).copy()
        if self.check_rep:
            X0, indices, counts = np.unique(X, return_inverse=True,return_counts=True,axis=0)
            if len(X0) != len(X):
                self.X = X0
                self.indices=indices
                self.max_rep=np.max(counts)
            else:  
                self.X=X
        else:
            self.X=X
        n_data=self.X.shape[0]
        #if self.n_data<300 and self.vecch:
        #    self.vecch=True if n_data>=100 else False
        #else:
        #    self.vecch=True if n_data>=300 else False
        self.n_data = n_data
        #if self.n_data>=1e4:
        #    self.nn_method = 'approx'
        #else:
        #self.nn_method = 'exact'
        self.m=min(self.m, self.n_data-1)
        if reset:
            self.reinit_all_layer(reset_lengthscale=True)
            self.imp=imputer(self.all_layer, self.block)
            (self.imp).sample(burnin=10)
            self.compute_r2()
        else:
            if (self.X[:, None] == origin_X).all(-1).any(-1).all():
                sub_idx=np.where((origin_X==self.X[:,None]).all(-1))[1]
                self.update_all_layer_smaller(sub_idx)
                self.imp=imputer(self.all_layer, self.block)
                (self.imp).sample(burnin=50)
            elif (origin_X[:, None] == self.X).all(-1).any(-1).all():
                sub_idx=np.where((self.X==origin_X[:,None]).all(-1))[1]
                self.update_all_layer_larger(sub_idx)
                self.imp=imputer(self.all_layer, self.block)
                (self.imp).sample(burnin=50)
            else:
                self.reinit_all_layer(reset_lengthscale=False)
                self.imp=imputer(self.all_layer, self.block)
                (self.imp).sample(burnin=200)
            self.compute_r2()

    def update_all_layer_larger(self, sub_idx):
        """Update **all_layer** attribute with new input and output when the original input is a subset of the new one.
        """
        global_in=(self.X).copy()
        In=(self.X).copy()
        mask=np.zeros(len(self.X),dtype=bool)
        mask[sub_idx]=True
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            num_kernel=len(layer)
            if l!=self.n_layer-1:
                Out=np.empty((len(In),num_kernel))
            for k in range(num_kernel):
                kernel=layer[k]
                if l!=self.n_layer-1:
                    #kernel.vecch, kernel.m, kernel.nn_method = self.vecch, self.m, self.nn_method
                    kernel.m = self.m
                    if kernel.vecch:
                        if kernel.connect is not None:
                            mu=cond_mean_vecch(In[~mask,:][:,kernel.input_dim], global_in[~mask,:][:,kernel.connect], kernel.input, kernel.global_input, kernel.output, kernel.scale, kernel.length, kernel.nugget, kernel.name, 50, kernel.nn_method)
                        else:
                            mu=cond_mean_vecch(In[~mask,:][:,kernel.input_dim], None, kernel.input, kernel.global_input, kernel.output, kernel.scale, kernel.length, kernel.nugget, kernel.name, 50, kernel.nn_method)
                    else: 
                        R=kernel.k_matrix()
                        L=np.linalg.cholesky(R)
                        Rinv_y=cho_solve((L, True), kernel.output, check_finite=False).flatten()
                        if kernel.connect is not None:
                            mu=cond_mean(In[~mask,:][:,kernel.input_dim],global_in[~mask,:][:,kernel.connect],kernel.input,kernel.global_input,Rinv_y,kernel.length,kernel.name)
                        else:
                            mu=cond_mean(In[~mask,:][:,kernel.input_dim],None,kernel.input,kernel.global_input,Rinv_y,kernel.length,kernel.name)
                    kernel.input=(In[:,kernel.input_dim]).copy()
                    Out[sub_idx,k]=kernel.output.flatten()
                    Out[~mask,k]=mu
                    kernel.output=Out[:,[k]].copy()
                    if kernel.connect is not None:
                        kernel.global_input=(global_in[:,kernel.connect]).copy()
                    if kernel.vecch:
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        kernel.max_rep = self.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                else:
                    kernel.rep=self.indices
                    #kernel.rep_sp=rep_sp(kernel.rep)
                    if kernel.rep is None:
                        kernel.input=(In[:,kernel.input_dim]).copy()
                    else:
                        kernel.input=(In[kernel.rep,:][:,kernel.input_dim]).copy()
                    if kernel.type=='gp':
                        if kernel.connect is not None:
                            if kernel.rep is None:
                                kernel.global_input=(global_in[:,kernel.connect]).copy()
                            else:
                                kernel.global_input=(global_in[kernel.rep,:][:,kernel.connect]).copy()
                        #kernel.vecch, kernel.m, kernel.nn_method = self.vecch, self.m, self.nn_method
                        kernel.m = self.m
                        if kernel.vecch:
                            if k == 0:
                                kernel.ord_nn(pointer=False)
                            else:
                                if len(kernel.length) == 1:
                                    found_match = False
                                    for j in range(k):
                                        if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                            kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=False)
                                            found_match = True
                                            break
                                    if not found_match:
                                        kernel.ord_nn(pointer=False)
                                else:
                                    found_match = False
                                    for j in range(k):
                                        if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                            kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=False)
                                            found_match = True
                                            break
                                    if not found_match:
                                        kernel.ord_nn(pointer=False)
                    kernel.output=(self.Y[:,[k]]).copy()
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        kernel.compute_cl()
            if l!=self.n_layer-1:
                In=(Out).copy()

    def update_all_layer_smaller(self, sub_idx):
        """Update **all_layer** attribute with new input and output when the new input is a subset of the original one.
        """
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            num_kernel=len(layer)
            for k in range(num_kernel):
                kernel=layer[k]
                if l==self.n_layer-1:
                    if kernel.rep is None:
                        kernel.input=kernel.input[sub_idx,:]
                        if self.indices is not None:
                            kernel.input=kernel.input[self.indices,:]
                    else:
                        kernel.input=np.concatenate([np.unique(kernel.input[kernel.rep==i,:],axis=0) for i in range(np.max(kernel.rep)+1)], axis=0)[sub_idx,:]
                        if self.indices is not None:
                            kernel.input=kernel.input[self.indices,:]
                    kernel.rep=self.indices
                    #kernel.rep_sp=rep_sp(kernel.rep)
                else:
                    kernel.input=kernel.input[sub_idx,:]
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            if kernel.rep is None:
                                kernel.global_input=(self.X[:,kernel.connect]).copy()
                            else:
                                kernel.global_input=(self.X[kernel.rep,:][:,kernel.connect]).copy()
                        else:
                            if l==0 and len(np.intersect1d(kernel.connect,kernel.input_dim))!=0:
                                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
                            kernel.global_input=(self.X[:,kernel.connect]).copy()
                    #kernel.vecch, kernel.m, kernel.nn_method = self.vecch, self.m, self.nn_method
                    kernel.m = self.m
                    if kernel.vecch:
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        kernel.max_rep = self.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                if l==self.n_layer-1:
                    kernel.output=(self.Y[:,[k]]).copy()
                else:
                    kernel.output=(kernel.output[sub_idx,:]).copy()
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        kernel.compute_cl()

    def reinit_all_layer(self, reset_lengthscale):
        """Reinitialise **all_layer** attribute with new input and output.
        Args: 
            reset_lengthscale (bool): whether to reset hyperparameter of the DGP emulator to the initial values.
        """
        global_in=self.X
        In=self.X
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            num_kernel=len(layer)
            if l!=self.n_layer-1:
                if np.shape(In)[1]==num_kernel:
                    Out=In
                elif np.shape(In)[1]>num_kernel:
                    if self.vecch or self.n_data>=500:
                        pca=NystromKPCA(n_components=num_kernel)
                        Out=pca.fit_transform(In)
                    else:
                        pca=KernelPCA(n_components=num_kernel, kernel='sigmoid')
                        Out=pca.fit_transform(In)
                else:
                    Out=np.concatenate((In, In[:,np.random.choice(np.shape(In)[1],num_kernel-np.shape(In)[1])]),1)
                Out=copy.copy(Out)
            for k in range(num_kernel):
                kernel=layer[k]
                if l==self.n_layer-1 and self.indices is not None:
                    kernel.rep=self.indices
                    #kernel.rep_sp=rep_sp(kernel.rep)
                if l==self.n_layer-1:
                    if kernel.rep is None:
                        kernel.input=In[:,kernel.input_dim]
                    else:
                        kernel.input=In[kernel.rep,:][:,kernel.input_dim]
                else:
                    kernel.input=In[:,kernel.input_dim]
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            if kernel.rep is None:
                                kernel.global_input=global_in[:,kernel.connect]
                            else:
                                kernel.global_input=global_in[kernel.rep,:][:,kernel.connect]
                        else:
                            if l==0 and len(np.intersect1d(kernel.connect,kernel.input_dim))!=0:
                                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
                            kernel.global_input=global_in[:,kernel.connect]
                    #kernel.vecch, kernel.m, kernel.nn_method = self.vecch, self.m, self.nn_method
                    kernel.m = self.m
                    if reset_lengthscale:
                        initial_hypers=kernel.para_path[0,:]
                        kernel.scale=initial_hypers[[0]]
                        kernel.length=initial_hypers[1:-1]
                        kernel.nugget=initial_hypers[[-1]]
                    if kernel.vecch:
                        compute_pointer = False
                        if l==self.n_layer-2:
                            linked_layer=self.all_layer[l+1]
                            linked_upper_kernels=[linked_kernel for linked_kernel in linked_layer if linked_kernel.input_dim is None or k in linked_kernel.input_dim]
                            if len(linked_upper_kernels)==1 and linked_upper_kernels[0].type=='likelihood' and linked_upper_kernels[0].exact_post_idx!=None:
                                idxx=np.where(linked_upper_kernels[0].input_dim == k)[0] if linked_upper_kernels[0].input_dim is not None else np.array([k])
                                if idxx in linked_upper_kernels[0].exact_post_idx:
                                    compute_pointer = True
                                    if self.indices is not None:
                                        kernel.max_rep = self.max_rep
                                        kernel.rep_hetero = self.indices
                        if k == 0:
                            kernel.ord_nn(pointer=compute_pointer)
                        else:
                            if len(kernel.length) == 1:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and len(layer[j].length) == 1:
                                        kernel.ord_nn(ord = layer[j].ord, NNarray = layer[j].NNarray, pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                            else:
                                found_match = False
                                for j in range(k):
                                    if np.array_equal(kernel.input_dim, layer[j].input_dim) and np.array_equal(kernel.connect, layer[j].connect) and np.array_equal(kernel.length, layer[j].length):
                                        kernel.ord_nn(ord = layer[j].ord.copy(), NNarray = layer[j].NNarray.copy(), pointer=compute_pointer)
                                        found_match = True
                                        break
                                if not found_match:
                                    kernel.ord_nn(pointer=compute_pointer)
                if l==self.n_layer-1:
                    kernel.output=self.Y[:,[k]]
                else:
                    kernel.output=Out[:,k].reshape((-1,1))
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        kernel.compute_cl()
            if l!=self.n_layer-1:
                In=copy.copy(Out)
       
    def train(self, N=500, ess_burn=10, disable=False):
        """Train the DGP model.

        Args:
            N (int): number of iterations for stochastic EM. Defaults to `500`.
            ess_burn (int, optional): number of burnin steps for the ESS-within-Gibbs
                at each I-step of the SEM. Defaults to `10`.
            disable (bool, optional): whether to disable the training progress bar. 
                Defaults to `False`.
        """
        pgb=trange(1,N+1,disable=disable)
        for i in pgb:
            #I-step           
            (self.imp).sample(burnin=ess_burn)
            if self.vecch and (self.N+i & (self.N+i-1)) == 0 and self.N+i > 1:
                (self.imp).update_ord_nn()
            #M-step
            for l in range(self.n_layer):
                for kernel in self.all_layer[l]:
                    if kernel.type=='gp':
                        if kernel.prior_name=='ref':
                            kernel.compute_cl()
                        if l!=0:
                            kernel.r2()
                        kernel.maximise()
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
        self.N += N

    def ptrain(self, N=500, ess_burn=10, disable=False, core_num=None):
        """Train the DGP model with parallel GP optimizations in each layer.

        Args:
            N (int): number of iterations for stochastic EM. Defaults to `500`.
            ess_burn (int, optional): number of burnin steps for the ESS-within-Gibbs
                at each I-step of the SEM. Defaults to `10`.
            disable (bool, optional): whether to disable the training progress bar. 
                Defaults to `False`.
            core_num (int, optional): the number of cores/workers to be used. Defaults to `None`. If not specified, 
                the number of cores is set to ``(max physical cores available - 1)``.
        """
        os_type = platform.system()
        if os_type in ['Darwin', 'Linux']:
            ctx._force_start_method('forkserver')
        total_cores = psutil.cpu_count(logical = False)
        if core_num is None:
            if self.vecch:
                core_num = total_cores//2
            else:
                core_num = total_cores - 1
        num_thread = total_cores // core_num

        def pmax(kernel):
            if kernel.type=='gp':
                if kernel.prior_name=='ref':
                    kernel.compute_cl()
                if kernel.vecch:
                    set_num_threads(num_thread)
                kernel.maximise()
            return kernel
        def pmax_r2(kernel):
            if kernel.type=='gp':
                if kernel.prior_name=='ref':
                    kernel.compute_cl()
                if kernel.vecch:
                    set_num_threads(num_thread)
                kernel.r2()
                kernel.maximise()
            return kernel
        
        pool = Pool(core_num)
        pgb=trange(1,N+1,disable=disable)
        for i in pgb:
            #I-step           
            (self.imp).sample(burnin=ess_burn)
            if self.vecch and (self.N+i & (self.N+i-1)) == 0 and self.N+i > 1:
                (self.imp).update_ord_nn()
            #M-step
            for l in range(self.n_layer):
                if l==0:
                    self.all_layer[l] = pool.map(pmax, self.all_layer[l])
                else:
                    self.all_layer[l] = pool.map(pmax_r2, self.all_layer[l])
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
        self.N += N
        pool.close()
        pool.join()
        pool.clear()

    def compute_r2(self):
        for l in range(1,self.n_layer):
            layer=self.all_layer[l]
            for kernel in layer:
                if kernel.type == 'gp':
                    kernel.r2(overwritten = True)

    def aggregate_r2(self,burnin=0.75,agg='median'):
        """Compute the aggregated R2 of all GP nodes on a DGP hierarchy.

        Args:
            burnin (float, optional): a value between 0 and 1 that indicates the percentage of 
                stored R2 values to be discarded for average R2 calculation. If this is not specified, 
                only the last 25% of R2 values are used. Defaults to 0.75.
            agg (str, optional): either 'median' or 'mean' that is used to aggregate the R2 values
                after discarding the first **burnin** percentage of the R2 sequences. Defaults to 'median'.

        Returns:
            list: a list of average R2 values that correspond to the DGP hierarchy.
        """
        if burnin<0 or burnin>1:
            raise Exception('burnin must be between 0 and 1.')
        r2_list = []
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            layer_r2_list = []
            for kernel in layer:
                if kernel.type == 'gp':
                    if kernel.R2 is None:
                        layer_r2_list.append(None)
                    else:
                        burnin_N=int(len(kernel.R2)*burnin)
                        if agg == 'mean':
                            layer_r2_list.append(np.mean(kernel.R2[burnin_N:,:],axis=0))
                        elif agg == 'median':
                            layer_r2_list.append(np.median(kernel.R2[burnin_N:,:],axis=0))
                        else:
                            raise Exception("agg must be either 'median' or 'mean'.")
                else:
                    layer_r2_list.append(None)
            r2_list.append(layer_r2_list)
        return r2_list

    def estimate(self,burnin=None):
        """Compute the point estimates of the DGP model parameters and output the trained DGP.

        Args:
            burnin (int, optional): the number of SEM iterations to be discarded for
                point estimate calculation. Must be smaller than the SEM iterations 
                implemented. If this is not specified, only the last 25% of iterations
                are used. Defaults to `None`.

        Returns:
            list: an updated list that represents the trained DGP hierarchy.
        """
        if burnin==None:
            self.burnin=int(self.N*(3/4))
        else:
            self.burnin=burnin
        final_struct=copy.deepcopy(self.all_layer)
        for l in range(len(final_struct)):
            for kernel in final_struct[l]:
                if kernel.type=='gp':
                    point_est=np.mean(kernel.para_path[self.burnin:,:],axis=0)
                    kernel.scale=np.atleast_1d(point_est[0])
                    kernel.length=np.atleast_1d(point_est[1:-1])
                    kernel.nugget=np.atleast_1d(point_est[-1])
        return final_struct

    def plot(self,layer_no,ker_no,width=4.,height=1.,ticksize=5.,labelsize=8.,hspace=0.1):
        """Plot the traces of model parameters of a particular GP node in the DGP hierarchy.

        Args:
            layer_no (int): the index of the interested layer.
            ker_no (int): the index of the interested GP in the layer specified by **layer_no**.
            width (float, optional): the overall plot width. Defaults to `4`.
            height (float, optional): the overall plot height. Defaults to `1`.
            ticksize (float, optional): the size of sub-plot ticks. Defaults to `5`.
            labelsize (float, optional): the font size of y labels. Defaults to `8`.
            hspace (float, optional): the space between sub-plots. Defaults to `0.1`.
        """
        kernel=self.all_layer[layer_no-1][ker_no-1]
        if kernel.type == 'gp':
            n_para=np.shape(kernel.para_path)[1]
            fig, axes = plt.subplots(n_para,figsize=(width,n_para*height), dpi= 100,sharex=True)
            fig.tight_layout()
            fig.subplots_adjust(hspace = hspace)
            for p in range(n_para):
                axes[p].plot(kernel.para_path[:,p])
                axes[p].tick_params(axis = 'both', which = 'major', labelsize = ticksize)
                if p==0:
                    axes[p].set_ylabel(r'$\sigma^2$',fontsize = labelsize)
                elif p==n_para-1:
                    axes[p].set_ylabel(r'$\eta$',fontsize = labelsize)
                else:
                    axes[p].set_ylabel(r'$\gamma_{%i}$' %p, fontsize = labelsize)
            plt.show()
        else:
            print('There is nothing to plot for a likelihood node, please choose a GP node instead.')

    