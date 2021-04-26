import numpy as np
from math import sqrt
from scipy.optimize import minimize
from functions import gp, link_gp

class kernel:
    def __init__(self, length, scale=1., nugget=1e-8, name='sexp', prior=np.array([0.3338,0.0835]), nugget_est=0, scale_est=0, input_dim=None, connect=None):
        #0.3338,0.0835
        self.length=length
        self.scale=np.atleast_1d(scale)
        self.nugget=np.atleast_1d(nugget)
        self.name=name
        self.prior=prior
        self.nugget_est=nugget_est
        self.scale_est=scale_est
        self.input_dim=input_dim
        self.connect=connect
        self.para_path=np.concatenate((self.scale,self.length,self.nugget))
        self.global_input=None
        self.input=None
        self.output=None

    def log_t(self):
        if self.nugget_est==1:
            log_theta=np.log(np.concatenate((self.length,self.nugget)))
        else:
            log_theta=np.log(self.length)
        return log_theta

    def update(self,log_theta):
        theta=np.exp(log_theta)
        if self.nugget_est==1:
            self.length=theta[0:-1]
            self.nugget=theta[-1]
        else:
            self.length=theta
        if self.scale_est==1:
            K=self.k_matrix()
            KinvY=np.linalg.solve(K,self.output)
            YKinvY=(self.output).T@KinvY
            new_scale=YKinvY/len(self.output)
            self.scale=new_scale.flatten()
            
    def k_matrix(self):
        n=len(self.output)
        if np.any(self.connect!=None):
            X=np.concatenate((self.input, self.global_input),1)
        else:
            X=self.input
        X_l=X/self.length
        if self.name=='sexp':
            L=np.sum(X_l**2,1).reshape([-1,1])
            dis2=np.abs(L-2*X_l@X_l.T+L.T)
            K=np.exp(-dis2)
        elif self.name=='matern2.5':
            X_l=np.expand_dims(X_l.T,axis=2)
            dis=np.abs(X_l-X_l.transpose([0,2,1]))
            K_1=np.prod(1+sqrt(5)*dis+5/3*dis**2,0)
            K_2=np.exp(-sqrt(5)*np.sum(dis,0))
            K=K_1*K_2
        return K+self.nugget*np.eye(n)

    def k_fod(self):
        n=len(self.output)
        if np.any(self.connect!=None):
            X=np.concatenate((self.input, self.global_input),1)
        else:
            X=self.input
        X_l=X/self.length
        
        X_li=np.expand_dims(X_l.T,axis=2)
        disi=np.abs(X_li-X_li.transpose([0,2,1]))
        if self.name=='sexp':
            dis2=np.sum(disi**2,axis=0,keepdims=True)
            K=np.exp(-dis2)
            if len(self.length)==1:
                fod=2*dis2*K
            else:
                fod=2*(disi**2)*K
        elif self.name=='matern2.5':
            K_1=np.prod(1+sqrt(5)*disi+5/3*disi**2,axis=0,keepdims=True)
            K_2=np.exp(-sqrt(5)*np.sum(disi,axis=0,keepdims=True))
            K=K_1*K_2
            coefi=(disi**2)*(1+sqrt(5)*disi)/(1+sqrt(5)*disi+5/3*disi**2)
            if len(self.length)==1:
                fod=5/3*np.sum(coefi,axis=0,keepdims=True)*K
            else:
                fod=5/3*coefi*K
        if self.nugget_est==1:
            nugget_fod=np.expand_dims(self.nugget*np.eye(n),0)
            fod=np.concatenate((fod,nugget_fod),axis=0)
        return fod
    
    def log_prior(self):
        lp=2*self.prior[0]*np.log(self.length)-self.prior[1]*self.length**2
        return np.sum(lp)

    def log_prior_fod(self):
        fod=2*(self.prior[0]-self.prior[1]*self.length**2)
        if self.nugget_est==1:
            fod=np.concatenate((fod,np.array([0])))
        return fod

    def llik(self,x):
        self.update(x)
        n=len(self.output)
        K=self.k_matrix()
        _,logdet=np.linalg.slogdet(K)
        KinvY=np.linalg.solve(K,self.output)
        YKinvY=(self.output).T@KinvY
        if self.scale_est==1:
            scale=YKinvY/n
            neg_llik=0.5*(logdet+n*np.log(scale))
        else:
            neg_llik=0.5*(logdet+YKinvY) 
        neg_llik=neg_llik.flatten()
        if np.any(self.prior!=None):
            neg_llik=neg_llik-self.log_prior()
        return neg_llik

    def llik_der(self,x):
        self.update(x)
        n=len(self.output)
        K=self.k_matrix()
        Kt=self.k_fod()
        KinvKt=np.linalg.solve(K,Kt)
        tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
        KinvY=np.linalg.solve(K,self.output)
        YKinvKtKinvY=((self.output).T@KinvKt@KinvY).flatten()
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if self.scale_est==1:
            YKinvY=(self.output).T@KinvY
            scale=(YKinvY/n).flatten()
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2
        if np.any(self.prior!=None):
            neg_St=neg_St-self.log_prior_fod()
        return neg_St

    def maximise(self, method='L-BFGS-B'):
        initial_theta_trans=self.log_t()
        re = minimize(self.llik, initial_theta_trans, method=method, jac=self.llik_der)
        if re.success!=True:
            re = minimize(self.llik, re.x, method='Nelder-Mead')
        self.add_to_path()
        
    def add_to_path(self):
        para=np.concatenate((self.scale,self.length,self.nugget))
        self.para_path=np.vstack((self.para_path,para))

    def gp_prediction(self,x,z):
        m,v=gp(x,z,self.input,self.global_input,self.output,self.scale,self.length,self.nugget,self.name)
        return m,v

    def linkgp_prediction(self,m,v,z):
        m,v=link_gp(m,v,z,self.input,self.global_input,self.output,self.scale,self.length,self.nugget,self.name)
        return m,v

def combine(*layers):
    all_layer=[]
    for layer in layers:
        all_layer.append(layer)
    return all_layer