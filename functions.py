from numba import jit
import numpy as np
from numpy.random import randn

@jit(nopython=True,cache=True)
def log_likelihood_func(y,cov):
    _,logdet=np.linalg.slogdet(cov)
    quad=np.sum(y*np.linalg.solve(cov,y))
    llik=-0.5*len(y)*np.log(2*np.pi)-0.5*(logdet+quad)
    return llik

@jit(nopython=True,cache=True)
def mvn(cov):
    d=len(cov)
    sn=randn(d)
    L=np.linalg.cholesky(cov)
    samp=np.sum(sn*L,axis=1)
    return samp

@jit(nopython=True,cache=True)
def k_one_matrix(X,length,nugget):
    X_l=X/length
    L=np.sum(X_l**2,1).reshape(len(X),1)
    dis=L-2*X_l@X_l.T+L.T
    K=np.exp(-dis)+nugget*np.identity(len(X))
    return K

@jit(nopython=True,cache=True)
def update_f(f,mean,nu,theta):
    fp=(f - mean)*np.cos(theta) + nu*np.sin(theta) + mean
    return fp

def Qlik(x,ker,w1,w2):
    ker.update(x)
    n=np.shape(w1)[0]
    K=ker.k_matrix(w1)
    H=np.ones(shape=[n,1])
    KinvH=np.linalg.solve(K,H)
    HKinvH=H.T@KinvH
    KinvY=np.linalg.solve(K,w2)
    HKinvY=H.T@KinvY
    b=HKinvY/HKinvH
    _,logdet=np.linalg.slogdet(K)
    R=w2-b*H
    KinvR=KinvY-b*KinvH
    RKinvR=R.T@KinvR

    if ker.scale_est==1:
        scale=RKinvR/(n-1)
        neg_qlik=0.5*(n-1)*np.log(2*np.pi)+0.5*(n-1)+0.5*(logdet+np.log(HKinvH)+(n-1)*np.log(scale))
    else:
        neg_qlik=0.5*(n-1)*np.log(2*np.pi)+0.5*(logdet+np.log(HKinvH)+RKinvR)
    neg_qlik=neg_qlik.flatten()

    if ker.prior_est==1:
        neg_qlik=neg_qlik-ker.log_prior()
    return neg_qlik

def Qlik_der(x,ker,w1,w2):
    ker.update(x)
    n=np.shape(w1)[0]
    K=ker.k_matrix(w1)
    H=np.ones(shape=[n,1])
    KinvH=np.linalg.solve(K,H)
    HKinvH=H.T@KinvH
    KinvY=np.linalg.solve(K,w2)
    HKinvY=H.T@KinvY
    b=HKinvY/HKinvH
    Kt=ker.k_fod(w1)
    KinvKt=np.linalg.solve(K,Kt)
    tr_KinvKt=np.trace(KinvKt,axis1=1, axis2=2)
    R=w2-b*H
    RKinvH=R.T@KinvH
    KinvR=KinvY-b*KinvH
    HKinvKtKinvH=H.T@KinvKt@KinvH
    RKinvKtKinvR=R.T@KinvKt@KinvR
    HKinvKtKinvY=H.T@KinvKt@KinvY
    bt=HKinvY/HKinvH**2*HKinvKtKinvH-HKinvKtKinvY/HKinvH
    P1=-0.5*tr_KinvKt+0.5*HKinvKtKinvH/HKinvH
    P2=RKinvH*bt+0.5*RKinvKtKinvR

    if ker.scale_est==1:
        RKinvR=R.T@KinvR
        scale=RKinvR/(n-1)
        neg_St=-P1-P2/scale
    else:
        neg_St=-P1-P2
    neg_St=neg_St.flatten()

    if ker.prior_est==1:
        neg_St=neg_St-ker.log_prior_fod()
    return neg_St