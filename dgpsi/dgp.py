import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import copy
from .imputation import imputer
from .functions import cmvn
from sklearn.decomposition import KernelPCA

class dgp:
    """
    Class that contains the deep GP hierarchy for stochastic imputation inference.

    Args:
        X (ndarray): a numpy 2d-array where each row is an input data point and 
            each column is an input dimension. 
        Y (list): a list of numpy 2d-arrays containing observed output data across the DGP structure. 
            If only one numpy 2d-array is supplied in the list, then it is assumed that no observations 
            are available for latent variables. In all other cases, the list needs to contain numpy 
            2d-arrays, one for each layer. Each 2d-array has it rows being output data points and
            columns being output dimensions (with the number of columns equals to the number of GP nodes
            in the corresponding layer). If there are missing values, those specific cells in the arrays 
            need to be filled by nan. The last array must have its number of columns equal to the 
            number of nodes (GPs) in the final layer.
        all_layer (list): a list contains L (the number of layers) sub-lists, each of which contains 
            the GPs defined by the kernel class in that layer. The sub-lists are placed in the list 
            in the same order of the specified DGP model.
        check_rep (bool, optional): whether to check the repetitions in the dataset, i.e., if one input
            position has multiple outputs. Defaults to True.

    Remark:
        This class is used for general DGP structures, which at least have some portions of internal I/O
            unobservable (i.e., there are latent variables contained in the structure). If you have fully
            observed internal I/O (i.e., the argument Y contains no nan), the DGP model reduces to linked
            GP model. In such a case, use lgp class for inference where one can have separate input/output
            training data for each GP. See lgp class for implementation details. 

    Examples:
        To build a list that represents a three-layer DGP with three GPs in the first two layers and
        one GP (i.e., only one dimensional output) in the final layer, do:
        >>>from kernel_class import kernel, combine
        >>>layer1, layer2, layer3=[],[],[]
        >>>for _ in range(3):
             layer1.append(kernel(length=np.array([1])))
        >>>for _ in range(3):
             layer2.append(kernel(length=np.array([1])))
        >>>layer3.append(kernel(length=np.array([1])))
        >>>all_layer=combine(layer1,layer2,layer3)       
    """

    def __init__(self, X, Y, all_layer, check_rep=True):
        self.Y=Y
        self.indices=None
        if check_rep:
            X0, indices = np.unique(X, return_inverse=True,axis=0)
            if len(X0) != len(X):
                if len(self.Y)!=1:
                    self.X=X
                else:
                    self.X = X0
                    self.indices=indices
            else:  
                self.X=X
        else:
            self.X=X
        self.n_layer=len(all_layer)
        self.all_layer=all_layer
        self.initialize()
        self.imp=imputer(self.all_layer)
        (self.imp).sample(burnin=10)
        self.N=0

    def initialize(self):
        """Initialise all_layer attribute for training.
        """
        if len(self.Y)==1:
            new_Y=[]
            for l in range(self.n_layer-1):
                num_kernel=len(self.all_layer[l])
                new_Y.append(np.full((len(self.X),num_kernel),np.nan))
            new_Y.append(self.Y[0])
            self.Y=copy.deepcopy(new_Y)
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
                kernel.missingness=copy.deepcopy(np.isnan(self.Y[l][:,k]))
                if l==self.n_layer-1 and self.indices is not None:
                    kernel.rep=self.indices
                if kernel.input_dim is not None:
                    if l==self.n_layer-1:
                        if kernel.type=='likelihood':
                            if kernel.name=='Poisson' and len(kernel.input_dim)!=1:
                                raise Exception('You need one and only one GP node to feed the ' + kernel.name + ' likelihood node.')
                            elif (kernel.name=='Hetero' or kernel.name=='NegBin') and len(kernel.input_dim)!=2:
                                raise Exception('You need two and only two GP nodes to feed the ' + kernel.name + ' likelihood node.')
                        kernel.last_layer_input=copy.deepcopy(In[:,kernel.input_dim])
                        if kernel.rep is None:
                            kernel.input=copy.deepcopy(kernel.last_layer_input[~kernel.missingness,:])
                        else:
                            kernel.input=copy.deepcopy(kernel.last_layer_input[kernel.rep,:][~kernel.missingness,:])
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
                        kernel.last_layer_input=copy.deepcopy(In)
                        if kernel.rep is None:
                            kernel.input=copy.deepcopy(kernel.last_layer_input[~kernel.missingness,:])
                        else:
                            kernel.input=copy.deepcopy(kernel.last_layer_input[kernel.rep,:][~kernel.missingness,:])
                    else:
                        kernel.input=copy.deepcopy(In)
                        kernel.input_dim=copy.deepcopy(np.arange(np.shape(In)[1]))
                if kernel.type=='gp':
                    if kernel.connect is not None:
                        if l==self.n_layer-1:
                            kernel.last_layer_global_input=copy.deepcopy(global_in[:,kernel.connect])
                            if kernel.rep is None:
                                kernel.global_input=copy.deepcopy(kernel.last_layer_global_input[~kernel.missingness,:])
                            else:
                                kernel.global_input=copy.deepcopy(kernel.last_layer_global_input[kernel.rep,:][~kernel.missingness,:])
                        else:
                            kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
                if not np.all(kernel.missingness):
                    if np.any(kernel.missingness):
                        if kernel.type=='likelihood':
                            print('The output of a likelihood node has to be fully observed. The training output of %i-th likelihood node in the final layer is partially missing.' % (k+1))
                            return
                        if l==self.n_layer-1:
                            samp=copy.deepcopy(self.Y[l][:,[k]])
                            kernel.output=copy.deepcopy(samp[~kernel.missingness,:])
                            #Out[~kernel.missingness,[k]]=copy.deepcopy(samp[~kernel.missingness,:].flatten())
                        else:
                            m,v=cmvn(kernel.input,kernel.global_input,self.Y[l][:,[k]],kernel.scale,kernel.length,kernel.nugget,kernel.name,kernel.missingness)
                            samp=copy.deepcopy(self.Y[l][:,[k]])
                            samp[kernel.missingness,0]=np.random.default_rng().multivariate_normal(mean=m,cov=v,check_valid='ignore')
                            kernel.output=copy.deepcopy(samp)
                            Out[:,[k]]=samp
                    else:
                        kernel.output=copy.deepcopy(self.Y[l][:,[k]])
                        if l!=self.n_layer-1:
                            Out[:,[k]]=self.Y[l][:,[k]]      
                else:
                    if kernel.type=='likelihood':
                        raise Exception('The output of a likelihood node has to be fully observed. The training output of %i-th likelihood node in the final layer is missing.' % (k+1))
                    if l==self.n_layer-1:
                        raise Exception('The output of GP nodes in the final layer cannot be completely missing. The training output of %i-th node in the final layer is fully missing.' % (k+1))
                    kernel.output=copy.deepcopy(Out[:,k].reshape((-1,1)))
            if l!=self.n_layer-1:
                In=copy.deepcopy(Out)
       
    def train(self, N=500, ess_burn=10, disable=False):
        """Train the DGP model.

        Args:
            N (int): number of iterations for stochastic EM. Defaults to 500.
            ess_burn (int, optional): number of burnin steps for the ESS-within-Gibbs
                at each I-step of the SEM. Defaults to 10.
            disable (bool, optional): whether to disable the training progress bar. 
                Defaults to False.
        """
        pgb=trange(1,N+1,disable=disable)
        for i in pgb:
            #I-step           
            (self.imp).sample(burnin=ess_burn)
            #M-step
            for l in range(self.n_layer):
                for kernel in self.all_layer[l]:
                    if kernel.type=='gp':
                        kernel.maximise()
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
        self.N += N

    def estimate(self,burnin=None):
        """Compute the point estimates of model parameters and output the trained DGP.

        Args:
            burnin (int, optional): the number of SEM iterations to be discarded for
                point estimate calculation. Must be smaller than the SEM iterations 
                implemented. If this is not specified, only the last 25% of iterations
                are used. Defaults to None.

        Returns:
            list: an updated list that represents the trained DGP hierarchy.
        """
        if burnin==None:
            burnin=int(self.N*(3/4))
        final_struct=copy.deepcopy(self.all_layer)
        for l in range(len(final_struct)):
            for kernel in final_struct[l]:
                if kernel.type=='gp':
                    point_est=np.mean(kernel.para_path[burnin:,:],axis=0)
                    kernel.scale=point_est[0]
                    kernel.length=point_est[1:-1]
                    kernel.nugget=point_est[-1]
        return final_struct

    def plot(self,layer_no,ker_no,width=4.,height=1.,ticksize=5.,labelsize=8.,hspace=0.1):
        """Plot the traces of model paramters of a particular GP node in the DGP hierarchy.

        Args:
            layer_no (int): the index of the interested layer
            ker_no (int): the index of the interested GP in the layer specified by layer_no
            width (float, optional): the overall plot width. Defaults to 4.
            height (float, optional): the overall plot height. Defaults to 1.
            ticksize (float, optional): the size of sub-plot ticks. Defaults to 5.
            labelsize (float, optional): the font size of y labels. Defaults to 8.
            hspace (float, optional): the space between sub-plots. Defaults to 0.1.
        """
        kernel=self.all_layer[layer_no-1][ker_no-1]
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

    