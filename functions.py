from numba import jit
import numpy as np
from numpy.random import randn

@jit(nopython=True,cache=True)
def log_likelihood_func(y,cov,scale,mean_prior,zero_mean):
    if zero_mean==1:
        cov=scale*cov
        _,logdet=np.linalg.slogdet(cov)
        quad=np.sum(y*np.linalg.solve(cov,y))
        llik=-0.5*len(y)*np.log(2*np.pi)-0.5*(logdet+quad)
    else:
        cov=scale*(cov+mean_prior)
        _,logdet=np.linalg.slogdet(cov)
        quad=np.sum(y*np.linalg.solve(cov,y))
        llik=-0.5*len(y)*np.log(2*np.pi)-0.5*(logdet+quad)
    return llik

@jit(nopython=True,cache=True)
def mvn(cov,scale,mean_prior,zero_mean):
    d=len(cov)
    sn=randn(d)
    if zero_mean==1:
        L=np.linalg.cholesky(scale*cov)
        samp=np.sum(sn*L,axis=1)
    else:
        #one_vec=np.ones(d)
        #RinvOne=np.linalg.solve(cov,one_vec)
        #coef=np.sum(one_vec*RinvOne)+1/mean_prior
        #mat=np.eye(d)-RinvOne/coef
        #U=np.linalg.cholesky(np.linalg.solve(cov,mat)/scale).T
        #samp=np.linalg.solve(U,sn)
        L=np.linalg.cholesky(scale*(cov+mean_prior))
        samp=np.sum(sn*L,axis=1)
    return samp

@jit(nopython=True,cache=True)
def k_one_matrix(X,length,nugget,name):
    if name=='sexp':
        n=len(X)
        X_l=X/length
        L=np.sum(X_l**2,1).reshape(n,1)
        dis=L-2*X_l@X_l.T+L.T
        K=np.exp(-dis)+nugget*np.identity(n)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        X_l=(X/length).T.reshape((d,n,1))
        L=X_l**2
        K1=np.ones((n,n))
        K2=np.zeros((n,n))
        for i in range(d):
            dis=L[i]-2*X_l[i]@X_l[i].T+L[i].T
            K1=K1*(1+np.sqrt(5*dis)+5/3*dis)
            K2=K2+np.sqrt(5*dis)
        K2=np.exp(-K2)
        K=K1*K2+nugget*np.identity(n)
    return K

@jit(nopython=True,cache=True)
def update_f(f,mean,nu,theta):
    fp=(f - mean)*np.cos(theta) + nu*np.sin(theta) + mean
    return fp

def Qlik(x,ker,w1,w2):
    ker.update(x)
    n=np.shape(w1)[0]
    K=ker.k_matrix(w1)
    if ker.zero_mean==0:
        _,logdet=np.linalg.slogdet(K+ker.mean_prior)
        KvinvY=np.linalg.solve(K+ker.mean_prior,w2)
        YKvinvY=w2.T@KvinvY
        if ker.scale_est==1:
            scale=YKvinvY/n
            neg_qlik=0.5*(logdet+n*np.log(scale))
        else:
            neg_qlik=0.5*(logdet+YKvinvY) 
        #_,logdet=np.linalg.slogdet(K)
        #KinvY=np.linalg.solve(K,w2)
        #YKinvY=w2.T@KinvY
        #H=np.ones(shape=[n,1])
        #KinvH=np.linalg.solve(K,H)
        #HKinvH=H.T@KinvH
        #HKinvY=H.T@KinvY
        #HKinvHv=HKinvH+1/ker.mean_prior
        #if ker.scale_est==1:
        #    scale=(YKinvY-HKinvY**2/HKinvHv)/n
        #    neg_qlik=0.5*(logdet+np.log(HKinvHv)+(n-1)*np.log(scale))
        #else:
        #    neg_qlik=0.5*(logdet+np.log(HKinvHv)+YKinvY-HKinvY**2/HKinvHv)
    else:
        _,logdet=np.linalg.slogdet(K)
        KinvY=np.linalg.solve(K,w2)
        YKinvY=w2.T@KinvY
        if ker.scale_est==1:
            scale=YKinvY/n
            neg_qlik=0.5*(logdet+n*np.log(scale))
        else:
            neg_qlik=0.5*(logdet+YKinvY) 
    neg_qlik=neg_qlik.flatten()

    if ker.prior_est==1:
        neg_qlik=neg_qlik-ker.log_prior()
    return neg_qlik

def Qlik_der(x,ker,w1,w2):
    ker.update(x)
    n=np.shape(w1)[0]
    K=ker.k_matrix(w1)
    Kt=ker.k_fod(w1)
    if ker.zero_mean==0:
        KvinvKt=np.linalg.solve(K+ker.mean_prior,Kt)
        tr_KvinvKt=np.trace(KvinvKt,axis1=1, axis2=2)
        KvinvY=np.linalg.solve(K+ker.mean_prior,w2)
        YKvinvKtKvinvY=w2.T@KvinvKt@KvinvY
        P1=-0.5*tr_KvinvKt
        P2=0.5*YKvinvKtKvinvY
        if ker.scale_est==1:
            YKvinvY=w2.T@KvinvY
            scale=YKvinvY/n
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2
        #KinvKt=np.linalg.solve(K,Kt)
        #tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
        #KinvY=np.linalg.solve(K,w2)
        #H=np.ones(shape=[n,1])
        #KinvH=np.linalg.solve(K,H)
        #HKinvH=H.T@KinvH
        #HKinvHv=HKinvH+1/ker.mean_prior
        #HKinvY=H.T@KinvY
        #HKinvKtKinvH=H.T@KinvKt@KinvH
        #HKinvKtKinvY=H.T@KinvKt@KinvY
        #YKinvKtKinvY=w2.T@KinvKt@KinvY
        #b=HKinvY/HKinvHv
        #P1=-0.5*tr_KinvKt+0.5*HKinvKtKinvH/HKinvHv
        #P2=0.5*YKinvKtKinvY+0.5*b**2*HKinvKtKinvH-b*HKinvKtKinvY
        #if ker.scale_est==1:
        #    YKinvY=w2.T@KinvY
        #    scale=(YKinvY-HKinvY**2/HKinvHv)/n
        #    neg_St=-P1-P2/scale
        #else:
        #    neg_St=-P1-P2
    else:
        KinvKt=np.linalg.solve(K,Kt)
        tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
        KinvY=np.linalg.solve(K,w2)
        YKinvKtKinvY=w2.T@KinvKt@KinvY
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if ker.scale_est==1:
            YKinvY=w2.T@KinvY
            scale=YKinvY/n
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2
    neg_St=neg_St.flatten()

    if ker.prior_est==1:
        neg_St=neg_St-ker.log_prior_fod()
    return neg_St

def linkgp(z,adj_sample,all_ker):
    for l in range(len(all_ker)):
        ker=all_ker[l]
        w1=adj_sample[l]
        w2=adj_sample[l+1]
        if l==0:
            m,v=gp(z,w1,w2,ker.scale,ker.length,ker.nugget,ker.name,ker.mean_prior,ker.zero_mean)
        else:
            m,v=link(m,v,w1,w2,ker.scale,ker.length,ker.nugget,ker.name,ker.mean_prior,ker.zero_mean)
    return m, v

@jit(nopython=True,cache=True)
def gp(z,w1,w2,scale,length,nugget,name,mean_prior,zero_mean):
    N=len(w1)
    M=len(z)
    m=np.empty((N,M))
    v=np.empty((N,M))
    for i in range(N):
        X=w1[i]
        y=w2[i]
        R=k_one_matrix(X,length,nugget,name)
        r=k_one_vec(X,z,length,name)
        if zero_mean==1:
            Rinv_r=np.linalg.solve(R,r)
            m[i,]=y.T@Rinv_r
            r_Rinv_r=np.sum(r*Rinv_r,axis=1)
            v[i,]=abs(scale*(1+nugget-r_Rinv_r))
        else:
            H=np.ones((len(R),1))
            Rinv_r=np.linalg.solve(R,r)
            Rinv_H=np.linalg.solve(R,H)
            yRinvH=np.sum(y*Rinv_H)
            HRinHv=np.sum(Rinv_H)+1/mean_prior
            b=yRinvH/HRinHv
            res=y-b
            m[i,]=res.T@Rinv_r+b
            r_Rinv_r=np.sum(r*Rinv_r,axis=1)
            r_Rinv_H=np.sum(r*Rinv_H,axis=1)
            v[i,]=abs(scale*(1+nugget-r_Rinv_r+(1-r_Rinv_H)**2/HRinHv))
    return m, v

@jit(nopython=True,cache=True)
def link(m,v,w1,w2,scale,length,nugget,name,mean_prior,zero_mean):
    
    return m,v

@jit(nopython=True,cache=True)
def k_one_vec(X,z,length,name):
    if name=='sexp':
        n=len(X)
        m=len(z)
        X_l=X/length
        z_l=z/length
        L_X=np.sum(X_l**2,1).reshape(n,1)
        L_z=np.sum(z_l**2,1).reshape(m,1)
        dis=L_X-2*X_l@z_l.T+L_z.T
        k=np.exp(-dis)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        m=len(z)
        X_l=(X/length).T.reshape((d,n,1))
        z_l=(z/length).T.reshape((d,m,1))
        L_X=X_l**2
        L_z=z_l**2
        k1=np.ones((n,m))
        k2=np.zeros((n,m))
        for i in range(d):
            dis=L_X[i]-2*X_l[i]@z_l[i].T+L_z[i].T
            k1=k1*(1+np.sqrt(5*dis)+5/3*dis)
            k2=k2+np.sqrt(5*dis)
        k2=np.exp(-k2)
        k=k1*k2
    return k