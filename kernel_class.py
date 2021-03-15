import numpy as np

class kernel:
    def __init__(self, length, scale=1., nugget=1e-6, name='sexp', prior=np.array([0.3338,0.0835]),scale_prior=np.array([1.,1.]),nugget_est=0, scale_est=0, prior_est=1, scale_prior_est=0, connect=0):
        #0.3338,0.0835
        self.length=length
        self.scale=np.atleast_1d(scale)
        self.scale_est=scale_est
        self.nugget=np.atleast_1d(nugget)
        self.nugget_est=nugget_est
        self.prior=prior
        self.scale_prior=scale_prior
        self.prior_est=prior_est
        self.scale_prior_est=scale_prior_est
        self.n_theta=len(length)
        self.name=name
        if nugget_est==1:
            self.n_theta=self.n_theta+1
        self.connect=connect
        self.global_input=[]

    def collect_para(self):
        para=np.concatenate((self.scale,self.length,self.nugget))
        return para

    def assign_point_para(self,para_data):
        point_para=np.mean(para_data,axis=0)
        self.scale=point_para[0]
        self.length=point_para[1:len(self.length)+1]
        self.nugget=point_para[len(self.length)+1]

    def log_t(self):
        if self.nugget_est==1:
            log_theta=(np.concatenate((np.log(self.length),np.log(self.nugget))))
        else:
            log_theta=np.log(self.length)
        return log_theta

    def update(self,log_theta):
        theta=np.exp(log_theta)
        if self.nugget_est==1:
            self.length=theta[0:len(self.length)]
            self.nugget=theta[len(self.length)]
        else:
            self.length=theta
        
    def k_matrix(self, X):
        n=np.shape(X)[0]
        if self.connect==1:
            X_l=X/self.length[:-1]
            global_input_l=self.global_input/self.length[-1]
            global_L=np.sum(global_input_l**2,1).reshape([-1,1])
            global_dis=np.abs(global_L-2*global_input_l@global_input_l.T+global_L.T)
        else:
            X_l=X/self.length
        if self.name=='sexp':
            L=np.sum(X_l**2,1).reshape([-1,1])
            dis=np.abs(L-2*X_l@X_l.T+L.T)
            if self.connect==1:
                dis=dis+global_dis
            K=np.exp(-dis)+self.nugget*np.eye(n)
        elif self.name=='matern2.5':
            X_l=np.expand_dims(X_l.T,axis=2)
            L=X_l**2
            dis=np.abs(L-2*X_l@X_l.transpose([0,2,1])+L.transpose([0,2,1]))
            if self.connect==1:
                global_dis=np.expand_dims(global_dis,axis=0)
                dis=np.vstack((dis,global_dis))
            K_1=np.prod(1+np.sqrt(5*dis)+5/3*dis,0)
            K_2=np.exp(-np.sum(np.sqrt(5*dis),0))
            K=K_1*K_2+self.nugget*np.eye(n)
        return K

    def k_fod(self,X):
        n=np.shape(X)[0]
        if self.connect==1:
            X_l=X/self.length[:-1]
            global_input_l=self.global_input/self.length[-1]
            global_L=np.sum(global_input_l**2,1).reshape([-1,1])
            global_dis=global_L-2*global_input_l@global_input_l.T+global_L.T
        else:
            X_l=X/self.length
        X_li=np.expand_dims(X_l.T,axis=2)
        Li=X_li**2
        disi=Li-2*X_li@X_li.transpose([0,2,1])+Li.transpose([0,2,1])
        if self.connect==1:
            global_dis=np.expand_dims(global_dis,axis=0)
            disi=np.vstack((disi,global_dis))
        if self.name=='sexp':
            K=np.exp(-np.sum(disi,0))
            K=np.expand_dims(K,axis=0)
            fod=2*disi*K
        elif self.name=='matern2.5':
            K_1=np.prod(1+np.sqrt(5*disi)+5/3*disi,0)
            K_2=np.exp(-np.sum(np.sqrt(5*disi),0))
            K=K_1*K_2
            K=np.expand_dims(K,axis=0)
            coefi=disi*(1+np.sqrt(5*disi))/(1+np.sqrt(5*disi)+5/3*disi)
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


def combine(*kernels):
    all_kernel=[]
    for kernel in kernels:
        all_kernel.append(kernel)
    return all_kernel 