import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import copy
from elliptical_slice import imputer
from sklearn.decomposition import KernelPCA
import time

class dgp:
    #main algorithm
    def __init__(self, X, Y, all_layer):
        self.X=X
        self.Y=Y
        self.n_layer=len(all_layer)
        self.all_layer=all_layer
        self.initialize()
        self.imp=imputer(self.all_layer)
        (self.imp).sample(burnin=50)
        self.N=0

    def initialize(self):
        global_in=self.X
        In=self.X
        for l in range(self.n_layer):
            layer=self.all_layer[l]
            num_kernel=len(layer)
            if l==self.n_layer-1:
                Out=self.Y
            else:
                if np.shape(In)[1]==num_kernel:
                    Out=In
                else:
                    pca=KernelPCA(n_components=num_kernel, kernel='sigmoid')
                    Out=pca.fit_transform(In)
            for k in range(num_kernel):
                kernel=layer[k]
                if np.any(kernel.input_dim!=None):
                    kernel.input=copy.deepcopy(In[:,kernel.input_dim])
                else:
                    kernel.input=copy.deepcopy(In)
                    kernel.input_dim=copy.deepcopy(np.arange(np.shape(In)[1]))
                kernel.output=copy.deepcopy(Out[:,k].reshape((-1,1)))
                if np.any(kernel.connect!=None):
                    kernel.global_input=copy.deepcopy(global_in[:,kernel.connect])
            In=Out

    def train(self, N=500, ess_burn=10, disable=False):
        #sub_burn>=1
        pgb=trange(1,N+1,ncols='70%',disable=disable)
        for i in pgb:
            #I-step           
            (self.imp).sample(burnin=ess_burn)
            #M-step
            for l in range(self.n_layer):
                for kernel in self.all_layer[l]:
                    kernel.maximise()
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
            #time.sleep(0.1) 
        self.N += N

    def estimate(self,burnin=None):
        if burnin==None:
            burnin=int(self.N/3)
        final_struct=copy.deepcopy(self.all_layer)
        for l in range(len(final_struct)):
            for kernel in final_struct[l]:
                point_est=np.mean(kernel.para_path[burnin:,:],axis=0)
                kernel.scale=point_est[0]
                kernel.length=point_est[1:-1]
                kernel.nugget=point_est[-1]
        return final_struct

    def plot(self,layer_no,ker_no,width=4,height=1,ticksize=5,labelsize=8,hspace=0.1):
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

    