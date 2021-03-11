import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import copy
from scipy.optimize import minimize
from functions import Qlik, Qlik_der, linkgp
from elliptical_slice import ess
from sklearn.decomposition import KernelPCA
import arviz as az

class dgp:
    #main algorithm
    def __init__(self, X, Y, all_kernel):
        self.X=X
        self.Y=Y
        self.layer=len(all_kernel)
        self.all_kernel=all_kernel
        self.para_path=[]
        self.burnin=[]
        for kernel in all_kernel:
             self.para_path.append(kernel.collect_para())
        self.lastmcmc=[]
        self.samples=[]
        self.final_kernel=[]
        self.cur_opt_iter=0
        self.last_opt_iter=0

    def train(self, N=400, burnin=300, sub_burn=20, method='L-BFGS-B',latent_ini='sigmoid'):
        #sub_burn>=1
        #initialisation
        self.burnin=burnin
        pgb=trange(1,N+1,ncols='70%')
        for i in pgb:
            #S-step
            all_kernel_old=copy.deepcopy(self.all_kernel)
            if not self.lastmcmc:      
                if np.shape(self.X)[1]==1:
                    new_ini=[self.X]*(self.layer-1) 
                else:
                    pca=KernelPCA(n_components=1, kernel=latent_ini)
                    new_ini=[pca.fit_transform(self.X)]*(self.layer-1) 
            else:
                new_ini=self.lastmcmc
            obj=ess(all_kernel_old,self.X,self.Y,new_ini)
            if not self.lastmcmc:
                samples=obj.sample_ess(N=1,burnin=100)
            else:
                samples=obj.sample_ess(N=1,burnin=sub_burn)
            self.lastmcmc=samples[1:-1]
            #M-step
            for l in range(self.layer):
                ker=copy.deepcopy(all_kernel_old[l])
                w1=np.squeeze(samples[l],axis=0)
                w2=np.squeeze(samples[l+1],axis=0)
                self.all_kernel[l]=self.optim(w1,w2,ker,method)
                pgb.set_description('Iteration %i: Layer %i' % (i,l+1))
                self.para_path[l]=np.vstack((self.para_path[l],self.all_kernel[l].collect_para()))
        pare_path_thinned=[t[burnin:] for t in self.para_path]
        final_kernel=copy.deepcopy(self.all_kernel)
        for l in range(self.layer):
            final_kernel[l].assign_point_para(pare_path_thinned[l])
        self.final_kernel=final_kernel
        self.cur_opt_iter+=1
        self.last_opt_iter=self.cur_opt_iter-1
    
    def update_final_kernel(self,burnin):
        self.burnin=burnin
        pare_path_thinned=[t[burnin:] for t in self.para_path]
        for l in range(self.layer):
            self.final_kernel[l].assign_point_para(pare_path_thinned[l])
        self.cur_opt_iter+=1
        self.last_opt_iter=self.cur_opt_iter-1

    def plot(self,ker_no):
        para_no=int(np.shape(self.para_path[ker_no])[1])
        for p in range(para_no):
            trace=self.para_path[ker_no][:,p]
            plt.figure()
            plt.plot(trace)
            plt.show()

    @staticmethod
    def optim(w1,w2,ker,method):
        n=np.shape(w1)[0]
        old_theta_trans=ker.log_t()
        re = minimize(Qlik, old_theta_trans, args=(ker,w1,w2), method=method, jac=Qlik_der)
        #res=method
        if re.success!=True:
            re = minimize(Qlik, re.x, args=(ker,w1,w2), method='Nelder-Mead')
            #res='Nelder-Mead'
        new_theta_trans=re.x
        ker.update(new_theta_trans)
        
        if ker.scale_est==1:
            K=ker.k_matrix(w1)
            if ker.zero_mean==0:
                H=np.ones(shape=[n,1])
                KinvH=np.linalg.solve(K,H)
                HKinvH=H.T@KinvH
                KinvY=np.linalg.solve(K,w2)
                HKinvY=H.T@KinvY
                YKinvY=w2.T@KinvY
                HKinvHv=HKinvH+1/ker.mean_prior
                if ker.scale_prior_est==1:
                    new_scale=(YKinvY-HKinvY**2/HKinvHv+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])
                else:
                    new_scale=(YKinvY-HKinvY**2/HKinvHv)/n
                ker.scale=new_scale.flatten()
            else:
                KinvY=np.linalg.solve(K,w2)
                YKinvY=w2.T@KinvY
                if ker.scale_prior_est==1:
                    new_scale=(YKinvY+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])
                else:
                    new_scale=YKinvY/n
                ker.scale=new_scale.flatten()

        #if re.success==True:
        #    res='Coverged!'
        #else:
        #    res='NoConverge'
        return ker

    def predict(self, z, N, burnin=0, method='sampling',ini='sigmoid'):
        if N!=0:
            if not self.lastmcmc:    
                if np.shape(self.X)[1]==1:
                    new_ini=[self.X]*(self.layer-1) 
                else:
                    pca=KernelPCA(n_components=1, kernel=ini)
                    new_ini=[pca.fit_transform(self.X)]*(self.layer-1) 
            else:
                new_ini=self.lastmcmc
            if self.final_kernel:
                obj=ess(self.final_kernel,self.X,self.Y,new_ini)
            else:
                obj=ess(self.all_kernel,self.X,self.Y,new_ini)
            samples=obj.sample_ess(N=N,burnin=1)
            if self.cur_opt_iter==0:
                if self.samples:
                    self.samples=[np.vstack((i,j)) for i,j in zip(self.samples,samples)]
                else:
                    self.samples=samples
            else:
                if self.cur_opt_iter-1==self.last_opt_iter:
                    self.samples=samples
                    self.last_opt_iter+=1
                elif self.cur_opt_iter==self.last_opt_iter:
                    self.samples=[np.vstack((i,j)) for i,j in zip(self.samples,samples)]
            self.lastmcmc=[t[-1] for t in samples[1:-1]]
        adj_sample=[t[burnin:] for t in self.samples]
        if self.final_kernel:
            mean,variance=linkgp(z,adj_sample,self.final_kernel)
        else:
            mean,variance=linkgp(z,adj_sample,self.all_kernel)
        print(f"se = {np.mean(np.array([az.mcse((mean[:,l,:]).flatten()) for l in range(np.shape(mean)[1])]))}")
        if method=='sampling':
            realisation=np.random.normal(mean,np.sqrt(variance))
            return np.squeeze(realisation)
        elif method=='mean_var':
            mu=np.mean(mean,axis=0)
            sigma2=np.mean((mean**2+variance),axis=0)-mu**2
            return mu.flatten(), sigma2.flatten()



    