import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import copy
from .imputation import imputer
from .kernel_class import kernel as ker
from .kernel_class import combine
from .functions import cond_mean
from sklearn.decomposition import KernelPCA
from scipy.linalg import cho_solve

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
            structure (for deterministic model emulation) with the number of GP nodes in the first layer equal 
            to the dimension of **X** is automatically constructed.
        check_rep (bool, optional): whether to check the repetitions in the dataset, i.e., if one input
            position has multiple outputs. Defaults to `True`.
        rff (bool, optional): whether to use random Fourier features to approximate the correlation matrices 
            during the imputation in training. Defaults to `False`.
        M (int, optional): the number of features to be used by random Fourier approximation. It is only used
            when **rff** is set to `True`. Defaults to `None`. If it is not specified, **M** is set to 
            ``max(100, ceil(sqrt(Data Size)*log(Data Size))))``.
        
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

    def __init__(self, X, Y, all_layer=None, check_rep=True, rff=False, M=None):
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
            X0, indices = np.unique(X, return_inverse=True,axis=0)
            if len(X0) != len(X):
                self.X = X0
                self.indices=indices
            else:  
                self.X=X
        else:
            self.X=X
        self.rff=rff
        if M is None:
            self.M=max(100, int(np.ceil(np.sqrt(len(self.X))*np.log(len(self.X)))))
        else:
            self.M=M
        if all_layer is None:
            D, Y_D=np.shape(self.X)[1], np.shape(self.Y)[1]
            layer1 = [ker(length=np.array([1.])) for _ in range(D)]
            layer2 = [ker(length=np.array([1.]),scale_est=True,connect=np.arange(D)) for _ in range(Y_D)]
            all_layer=combine(layer1,layer2)
        self.all_layer=all_layer
        self.n_layer=len(self.all_layer)
        self.initialize()
        self.imp=imputer(self.all_layer)
        (self.imp).sample(burnin=10)
        self.N=0
        self.burnin=None

    def initialize(self):
        """Initialise all_layer attribute for training.
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
                    pca=KernelPCA(n_components=num_kernel, kernel='sigmoid')
                    Out=pca.fit_transform(In)
                else:
                    Out=np.concatenate((In, In[:,np.random.choice(np.shape(In)[1],num_kernel-np.shape(In)[1])]),1)
                Out=copy.deepcopy(Out)
            for k in range(num_kernel):
                kernel=layer[k]
                if l==self.n_layer-1 and self.indices is not None:
                    kernel.rep=self.indices
                if kernel.input_dim is not None:
                    if l==self.n_layer-1:
                        if kernel.type=='likelihood':
                            if kernel.name=='Poisson' and len(kernel.input_dim)!=1:
                                raise Exception('You need one and only one GP node to feed the ' + kernel.name + ' likelihood node.')
                            elif (kernel.name=='Hetero' or kernel.name=='NegBin') and len(kernel.input_dim)!=2:
                                raise Exception('You need two and only two GP nodes to feed the ' + kernel.name + ' likelihood node.')
                        if kernel.rep is None:
                            kernel.input=copy.deepcopy(In[:,kernel.input_dim])
                        else:
                            kernel.input=copy.deepcopy(In[kernel.rep,:][:,kernel.input_dim])
                    else:
                        kernel.input=copy.deepcopy(In[:,kernel.input_dim])
                else:
                    if l==self.n_layer-1:
                        kernel.input_dim=copy.deepcopy(np.arange(np.shape(In)[1]))
                        if kernel.type=='likelihood':
                            if kernel.name=='Poisson' and len(kernel.input_dim)!=1:
                                raise Exception('You need one and only one GP node to feed the ' + kernel.name + ' likelihood node.')
                            elif (kernel.name=='Hetero' or kernel.name=='NegBin') and len(kernel.input_dim)!=2:
                                raise Exception('You need two and only two GP nodes to feed the ' + kernel.name + ' likelihood node.')
                        if kernel.rep is None:
                            kernel.input=copy.deepcopy(In)
                        else:
                            kernel.input=copy.deepcopy(In[kernel.rep,:])
                    else:
                        kernel.input=copy.deepcopy(In)
                        kernel.input_dim=copy.deepcopy(np.arange(np.shape(In)[1]))
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            if kernel.rep is None:
                                kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
                            else:
                                kernel.global_input=copy.deepcopy(global_in[kernel.rep,:][:,kernel.connect])
                        else:
                            if l==0 and len(np.intersect1d(kernel.connect,kernel.input_dim))!=0:
                                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
                            kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
                    kernel.rff, kernel.M = self.rff, self.M
                    kernel.D=np.shape(kernel.input)[1]
                    if kernel.connect is not None:
                        kernel.D+=len(kernel.connect)
                    if kernel.rff:
                        kernel.sample_basis()
                if l==self.n_layer-1:
                    kernel.output=copy.deepcopy(self.Y[:,[k]])
                else:
                    kernel.output=copy.deepcopy(Out[:,k].reshape((-1,1)))
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        p=np.shape(kernel.input)[1]
                        if kernel.global_input is not None:
                            p+=np.shape(kernel.global_input)[1]
                        b=1/len(kernel.output)**(1/p)*(kernel.prior_coef+p)
                        kernel.prior_coef=np.concatenate((kernel.prior_coef, b))
                        kernel.compute_cl()
            if l!=self.n_layer-1:
                In=copy.deepcopy(Out)

    def update_xy(self,X,Y):
        """Update the trained DGP emulator with new input and output data without changing the hyperparameter values.

        Args:
            X (ndarray): a numpy 2d-array where each row is an input data point and 
                each column is an input dimension. 
            Y (ndarray): a numpy 2d-arrays containing observed output data. 
                The 2d-array has it rows being output data points and columns being output dimensions 
                (with the number of columns equals to the number of GP nodes in the final layer). 
        """
        self.Y=Y
        if isinstance(self.Y, list):
            if len(self.Y)==1:
                self.Y=self.Y[0]
            else:
                raise Exception('Y has to be a numpy 2d-array rather than a list. The list version of Y (for linked emulation) has been reduced. Please use the dedicated lgp class for linked emulation.')
        if (self.Y).ndim==1 or X.ndim==1:
            raise Exception('The input and output data have to be numpy 2d-arrays.')
        self.indices=None
        origin_X=(self.X).copy()
        if self.check_rep:
            X0, indices = np.unique(X, return_inverse=True,axis=0)
            if len(X0) != len(X):
                self.X = X0
                self.indices=indices
            else:  
                self.X=X
        else:
            self.X=X
        if (self.X[:, None] == origin_X).all(-1).any(-1).all():
            sub_idx=np.where((origin_X==self.X[:,None]).all(-1))[1]
            self.update_all_layer_smaller(sub_idx)
            self.imp=imputer(self.all_layer)
            (self.imp).sample(burnin=50)
        elif (origin_X[:, None] == self.X).all(-1).any(-1).all():
            sub_idx=np.where((self.X==origin_X[:,None]).all(-1))[1]
            self.update_all_layer_larger(sub_idx)
            self.imp=imputer(self.all_layer)
            (self.imp).sample(burnin=50)
        else:
            self.reinit_all_layer()
            self.imp=imputer(self.all_layer)
            (self.imp).sample(burnin=200)

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
                else:
                    kernel.rep=self.indices
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
                        kernel.input=np.concatenate([np.unique(kernel.input[kernel.rep==i,:],axis=0) for i in range(np.max(kernel.rep))], axis=0)[sub_idx,:]
                        if self.indices is not None:
                            kernel.input=kernel.input[self.indices,:]
                    kernel.rep=self.indices
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
                if l==self.n_layer-1:
                    kernel.output=(self.Y[:,[k]]).copy()
                else:
                    kernel.output=(kernel.output[sub_idx,:]).copy()
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        kernel.compute_cl()

    def reinit_all_layer(self):
        """Reinitialise **all_layer** attribute with new input and output.
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
                    pca=KernelPCA(n_components=num_kernel, kernel='sigmoid')
                    Out=pca.fit_transform(In)
                else:
                    Out=np.concatenate((In, In[:,np.random.choice(np.shape(In)[1],num_kernel-np.shape(In)[1])]),1)
                Out=copy.deepcopy(Out)
            for k in range(num_kernel):
                kernel=layer[k]
                if l==self.n_layer-1 and self.indices is not None:
                    kernel.rep=self.indices
                if l==self.n_layer-1:
                    if kernel.rep is None:
                        kernel.input=copy.deepcopy(In[:,kernel.input_dim])
                    else:
                        kernel.input=copy.deepcopy(In[kernel.rep,:][:,kernel.input_dim])
                else:
                    kernel.input=copy.deepcopy(In[:,kernel.input_dim])
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            if kernel.rep is None:
                                kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
                            else:
                                kernel.global_input=copy.deepcopy(global_in[kernel.rep,:][:,kernel.connect])
                        else:
                            if l==0 and len(np.intersect1d(kernel.connect,kernel.input_dim))!=0:
                                raise Exception('The local input and global input should not have any overlap. Change input_dim or connect so they do not have any common indices.')
                            kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
                if l==self.n_layer-1:
                    kernel.output=copy.deepcopy(self.Y[:,[k]])
                else:
                    kernel.output=copy.deepcopy(Out[:,k].reshape((-1,1)))
                if kernel.type=='gp':
                    if kernel.prior_name=='ref':
                        kernel.compute_cl()
            if l!=self.n_layer-1:
                In=copy.deepcopy(Out)
       
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
            #M-step
            for l in range(self.n_layer):
                for kernel in self.all_layer[l]:
                    if kernel.type=='gp':
                        if kernel.prior_name=='ref':
                            kernel.compute_cl()
                        kernel.maximise()
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
        self.N += N

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
                    kernel.scale=point_est[0]
                    kernel.length=point_est[1:-1]
                    kernel.nugget=point_est[-1]
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

    