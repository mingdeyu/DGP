import numpy as np
from elliptical_slice import ess
from tqdm.notebook import trange, tqdm
import copy
from functions import Qlik, Qlik_der
from scipy.optimize import minimize

class dgp:
    #main algorithm
    def __init__(self, X, Y, layer, all_kernel):
        self.X=X
        self.Y=Y
        self.layer=layer
        self.all_kernel=all_kernel
        self.para_path=[]
        for kernel in all_kernel:
             self.para_path.append(kernel.collect_para())
        self.lastmcmc=[]

    def train(self, N, burnin=100, sub_burn=100, method='BFGS'):
        #sub_burn>=1
        #initialisation
        pgb=trange(1,N+1,ncols='70%')
        for i in pgb:
            #S-step
            all_kernel_old=copy.deepcopy(self.all_kernel)
            if not self.lastmcmc:    
                hidden=np.linspace(np.min(self.Y),np.max(self.Y), num=len(self.Y))   
                new_ini=[hidden]*(self.layer-1)    
            else:
                new_ini=self.lastmcmc
            obj=ess(all_kernel_old,self.X,self.Y,new_ini)
            samples=obj.sample_ess(N=1,burnin=sub_burn)
            self.lastmcmc=samples
            #M-step
            for l in range(self.layer):
                ker=copy.deepcopy(all_kernel_old[l])
                w1=np.squeeze(samples[l])
                w2=np.squeeze(samples[l+1])
                self.all_kernel[l],res=self.optim(w1,w2,ker,method)
                pgb.set_description('Iteration %i: Layer %i, %s' % (i,l+1,res))
                self.para_path[l]=np.vstack((self.para_path[l],self.all_kernel[l].collect_para()))
        return [t[burnin:] for t in self.para_path]
    
    @staticmethod
    def optim(w1,w2,ker,method):
        n=np.shape(w1)[0]
        old_theta_trans=ker.log_t()
        re = minimize(Qlik, old_theta_trans, args=(ker,w1,w2), method=method, jac=Qlik_der)
        if re.success!=True:
            re = minimize(Qlik, old_theta_trans, args=(ker,w1,w2), method='Nelder-Mead')
        new_theta_trans=re.x
        ker.update(new_theta_trans)

        K=ker.k_matrix(w1)
        H=np.ones(shape=[n,1])
        KinvH=np.linalg.solve(K,H)
        HKinvH=H.T@KinvH
        KinvY=np.linalg.solve(K,w2)
        HKinvY=H.T@KinvY
        new_b=HKinvY/HKinvH
        ker.mean=new_b.flatten()

        if ker.scale_est==1:
            R=w2-new_b*H
            KinvR=KinvY-new_b*KinvH
            RKinvR=R.T@KinvR
            new_scale=RKinvR/(n-1)
            ker.scale=new_scale.flatten()
        if re.success==True:
            res='Coverged!'
        else:
            res='NoConverge'

        return ker,res

    