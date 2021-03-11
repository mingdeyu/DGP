from numba import jit, vectorize, float64
import numpy as np
from math import erf, exp, sqrt, pi
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
    sn=randn(d,1)
    if zero_mean==1:
        L=np.linalg.cholesky(scale*cov)
        samp=(L@sn).flatten()
    else:
        #one_vec=np.ones(d)
        #RinvOne=np.linalg.solve(cov,one_vec)
        #coef=np.sum(RinvOne)+1/mean_prior
        #mat=np.eye(d)-RinvOne/coef
        #U=np.linalg.cholesky(np.linalg.solve(cov,mat)/scale).T
        #samp=np.linalg.solve(U,sn)
        L=np.linalg.cholesky(scale*(cov+mean_prior))
        samp=(L@sn).flatten()
    return samp

@jit(nopython=True,cache=True)
def k_one_matrix(X,length,nugget,name):
    if name=='sexp':
        n=len(X)
        X_l=X/length
        L=np.expand_dims(np.sum(X_l**2,axis=1),axis=1)
        dis=L-2*X_l@X_l.T+L.T
        K=np.exp(-dis)+nugget*np.identity(n)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        X_l=np.expand_dims((X/length).T,axis=2)
        L=X_l**2
        #K=np.ones((n,n))
        K1=np.ones((n,n))
        K2=np.zeros((n,n))
        for i in range(d):
            dis=np.abs(L[i]-2*X_l[i]@X_l[i].T+L[i].T)
            K1*=(1+np.sqrt(5*dis)+5/3*dis)
            K2+=np.sqrt(5*dis)
        K2=np.exp(-K2)
        K=K1*K2+nugget*np.identity(n)
            #K*=(1+np.sqrt(5*dis)+5/3*dis)*np.exp(-np.sqrt(5*dis))
        #K=K+nugget*np.identity(n)
    return K

@jit(nopython=True,cache=True)
def update_f(f,mean,nu,theta):
    fp=(f - mean)*np.cos(theta) + (nu - mean)*np.sin(theta) + mean
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
            if ker.scale_prior_est==1:
                scale=(YKvinvY+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])
                neg_qlik=0.5*(logdet+(n+2+2*ker.scale_prior[0])*np.log(scale))
            else:
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
        #    neg_qlik=0.5*(logdet+np.log(HKinvHv)+n*np.log(scale))
        #else:
        #    neg_qlik=0.5*(logdet+np.log(HKinvHv)+YKinvY-HKinvY**2/HKinvHv)
    else:
        _,logdet=np.linalg.slogdet(K)
        KinvY=np.linalg.solve(K,w2)
        YKinvY=w2.T@KinvY
        if ker.scale_est==1:
            if ker.scale_prior_est==1:
                scale=(YKinvY+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])
                neg_qlik=0.5*(logdet+(n+2+2*ker.scale_prior[0])*np.log(scale))
            else:
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
        YKvinvKtKvinvY=(w2.T@KvinvKt@KvinvY).flatten()
        P1=-0.5*tr_KvinvKt
        P2=0.5*YKvinvKtKvinvY
        if ker.scale_est==1:
            YKvinvY=w2.T@KvinvY
            if ker.scale_prior_est==1:
                scale=((YKvinvY+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])).flatten()
            else:
                scale=(YKvinvY/n).flatten()
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
        YKinvKtKinvY=(w2.T@KinvKt@KinvY).flatten()
        P1=-0.5*tr_KinvKt
        P2=0.5*YKinvKtKinvY
        if ker.scale_est==1:
            YKinvY=w2.T@KinvY
            if ker.scale_prior_est==1:
                scale=((YKinvY+2*ker.scale_prior[1])/(n+2+2*ker.scale_prior[0])).flatten()
            else:
                scale=(YKinvY/n).flatten()
            neg_St=-P1-P2/scale
        else:
            neg_St=-P1-P2
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
    X=w1[0]
    R=k_one_matrix(X,length,nugget,name)
    r=k_one_vec(X,z,length,name)
    if zero_mean==1:
        Rinv_r=np.linalg.solve(R,r)
        r_Rinv_r=np.sum(r*Rinv_r,axis=0)
        v=np.ones((N,1))*np.abs(scale*(1+nugget-r_Rinv_r))
    else:
        H=np.ones((len(R),1))
        Rinv_r=np.linalg.solve(R,r)
        Rinv_H=np.linalg.solve(R,H)
        HRinvHv=np.sum(Rinv_H)+1/mean_prior
        r_Rinv_r=np.sum(r*Rinv_r,axis=0)
        r_Rinv_H=np.sum(r*Rinv_H,axis=0)
        v=np.ones((N,1))*np.abs(scale*(1+nugget-r_Rinv_r+(1-r_Rinv_H)**2/HRinvHv))
    for i in range(N):
        y=w2[i]
        if zero_mean==1:
            m[i,]=y.T@Rinv_r
        else:
            yRinvH=y.T@Rinv_H
            b=yRinvH/HRinvHv
            res=y-b
            m[i,]=res.T@Rinv_r+b         
    m=np.expand_dims(m,axis=2)
    v=np.expand_dims(v,axis=2)
    return m, v

@jit(nopython=True,cache=True)
def link(m,v,w1,w2,scale,length,nugget,name,mean_prior,zero_mean):
    N=np.shape(m)[0]
    M=np.shape(m)[1]
    m_new=np.empty((N,M,1))
    v_new=np.empty((N,M,1))
    for i in range(N):
        X=w1[i]
        y=w2[i] 
        R=k_one_matrix(X,length,nugget,name)
        Rinv_y=np.linalg.solve(R,y)
        if zero_mean==0:
            H=np.ones((len(R),1))
            Rinv_H=np.linalg.solve(R,H)
            yRinvH=np.sum(y*Rinv_H)
            HRinvHv=np.sum(Rinv_H)+1/mean_prior
            b=yRinvH/HRinvHv
            Rinv_res=Rinv_y-b*Rinv_H
        for j in range(M):
            z_m=m[i,j]
            z_v=v[i,j]
            I,J=IJ(X,z_m,z_v,length,name)
            tr_RinvJ=np.trace(np.linalg.solve(R,J))
            if zero_mean==1:
                IRinv_y=np.sum(I*Rinv_y)
                m_new[i,j,]=IRinv_y
                v_new[i,j,]=np.abs(Rinv_y.T@J@Rinv_y-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
            else:
                HRinvI=np.sum(I*Rinv_H)
                HRinvJRinvH=Rinv_H.T@J@Rinv_H
                IRinv_res=np.sum(I*Rinv_res)
                m_new[i,j,]=b+IRinv_res
                v_new[i,j,]=np.abs(Rinv_res.T@J@Rinv_res-IRinv_res**2+scale*(1+nugget-tr_RinvJ+(1-2*HRinvI+HRinvJRinvH)/HRinvHv))
    
    return m_new,v_new

@jit(nopython=True,cache=True)
def k_one_vec(X,z,length,name):
    if name=='sexp':
        n=len(X)
        m=len(z)
        X_l=X/length
        z_l=z/length
        L_X=np.expand_dims(np.sum(X_l**2,axis=1),axis=1)
        L_z=np.expand_dims(np.sum(z_l**2,axis=1,),axis=1)
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
            dis=np.abs(L_X[i]-2*X_l[i]@z_l[i].T+L_z[i].T)
            k1*=(1+np.sqrt(5*dis)+5/3*dis)
            k2+=np.sqrt(5*dis)
        k2=np.exp(-k2)
        k=k1*k2
    return k

@jit(nopython=True,cache=True)
def IJ(X,z_m,z_v,length,name):
    n=np.shape(X)[0]
    d=np.shape(X)[1]
    if name=='sexp':
        X_z=X-z_m
        I=np.ones((n,1))
        J=np.ones((n,n))
        X_z=X_z.T.reshape((d,n,1))
        for i in range(d):
            I*=1/np.sqrt(1+2*z_v[i]/length[i]**2)*np.exp(-X_z[i]**2/(2*z_v[i]+length[i]**2))
            L_X_z=X_z[i]**2
            cross_L_X_z=X_z[i]@X_z[i].T
            dis1=L_X_z+2*cross_L_X_z+L_X_z.T
            dis2=L_X_z-2*cross_L_X_z+L_X_z.T
            J*=1/np.sqrt(1+4*z_v[i]/length[i]**2)*np.exp(-dis1/(2*length[i]**2+8*z_v[i])-dis2/(2*length[i]**2))
    elif name=='matern2.5':
        zX=z_m-X
        muA=(zX-sqrt(5)*z_v/length).T.reshape((d,n,1))
        muB=(zX+sqrt(5)*z_v/length).T.reshape((d,n,1))
        zX=zX.T.reshape((d,n,1))
        X=X.T.reshape((d,n,1))
        I=np.ones((n,1))
        J=np.ones((n**2,1))
        for i in range(d):    
            if z_v[i]!=0:
                I*=np.exp((5*z_v[i]-2*sqrt(5)*length[i]*zX[i])/(2*length[i]**2))* \
                    ((1+sqrt(5)*muA[i]/length[i]+5*(muA[i]**2+z_v[i])/(3*length[i]**2))*pnorm(muA[i]/sqrt(z_v[i]))+ \
                    (sqrt(5)+(5*muA[i])/(3*length[i]))*sqrt(0.5*z_v[i]/pi)/length[i]*np.exp(-0.5*muA[i]**2/z_v[i]))+ \
                   np.exp((5*z_v[i]+2*sqrt(5)*length[i]*zX[i])/(2*length[i]**2))* \
                    ((1-sqrt(5)*muB[i]/length[i]+5*(muB[i]**2+z_v[i])/(3*length[i]**2))*pnorm(-muB[i]/sqrt(z_v[i]))+ \
                    (sqrt(5)-(5*muB[i])/(3*length[i]))*sqrt(0.5*z_v[i]/pi)/length[i]*np.exp(-0.5*muB[i]**2/z_v[i]))
                X1_rep=np.ones((1,n))*X[i]
                X1_vec=X1_rep.reshape((n**2,1))
                X2_rep=np.ones((n,1))*X[i].T
                X2_vec=X2_rep.reshape((n**2,1))
                J*=Jd(X1_vec,X2_vec,z_m[i],z_v[i],length[i])
            else:
                Id=(1+sqrt(5)*np.abs(zX[i])/length[i]+5*zX[i]**2/(3*length[i]**2))*np.exp(-sqrt(5)*np.abs(zX[i])/length[i])
                I*=Id
                J*=(Id@Id.T).reshape((n**2,1))
        J=J.reshape((n,n))
    return I,J

@vectorize([float64(float64)],nopython=True,cache=True)
def pnorm(x):
    return 0.5*(1+erf(x/sqrt(2)))    

@vectorize([float64(float64,float64,float64,float64,float64)],nopython=True,cache=True)
def Jd(X1,X2,z_m,z_v,length):
    if X1<X2:
        x1=X1
        x2=X2
    else:
        x1=X2
        x2=X1
    E30=1+(25*x1**2*x2**2-3*sqrt(5)*(3*length**3+5*length*x1*x2)*(x1+x2)+15*length**2*(x1**2+x2**2+3*x1*x2))/(9*length**4)
    E31=(18*sqrt(5)*length**3+15*sqrt(5)*length*(x1**2+x2**2)-(75*length**2+50*x1*x2)*(x1+x2)+60*sqrt(5)*length*x1*x2)/(9*length**4)
    E32=5*(5*x1**2+5*x2**2+15*length**2-9*sqrt(5)*length*(x1+x2)+20*x1*x2)/(9*length**4)
    E33=10*(3*sqrt(5)*length-5*x1-5*x2)/(9*length**4)
    E34=25/(9*length**4)
    muC=z_m-2*sqrt(5)*z_v/length
    E3A31=E30+muC*E31+(muC**2+z_v)*E32+(muC**3+3*z_v*muC)*E33+(muC**4+6*z_v*muC**2+3*z_v**2)*E34
    E3A32=E31+(muC+x2)*E32+(muC**2+2*z_v+x2**2+muC*x2)*E33+(muC**3+x2**3+x2*muC**2+muC*x2**2+3*z_v*x2+5*z_v*muC)*E34
    P1=exp((10*z_v+sqrt(5)*length*(x1+x2-2*z_m))/length**2)*(0.5*E3A31*(1+erf((muC-x2)/sqrt(2*z_v)))+\
        E3A32*sqrt(0.5*z_v/pi)*exp(-0.5*(x2-muC)**2/z_v))
    
    E40=1+(25*x1**2*x2**2+3*sqrt(5)*(3*length**3-5*length*x1*x2)*(x2-x1)+15*length**2*(x1**2+x2**2-3*x1*x2))/(9*length**4)
    E41=5*(3*sqrt(5)*length*(x2**2-x1**2)+3*length**2*(x1+x2)-10*x1*x2*(x1+x2))/(9*length**4)
    E42=5*(5*x1**2+5*x2**2-3*length**2-3*sqrt(5)*length*(x2-x1)+20*x1*x2)/(9*length**4)
    E43=-50*(X1+X2)/(9*length**4)
    E44=25/(9*length**4)
    E4A41=E40+z_m*E41+(z_m**2+z_v)*E42+(z_m**3+3*z_v*z_m)*E43+(z_m**4+6*z_v*z_m**2+3*z_v**2)*E44
    E4A42=E41+(z_m+x1)*E42+(z_m**2+2*z_v+x1**2+z_m*x1)*E43+(z_m**3+x1**3+x1*z_m**2+z_m*x1**2+3*z_v*x1+5*z_v*z_m)*E44
    E4A43=E41+(z_m+x2)*E42+(z_m**2+2*z_v+x2**2+z_m*x2)*E43+(z_m**3+x2**3+x2*z_m**2+z_m*x2**2+3*z_v*x2+5*z_v*z_m)*E44
    P2=exp(-sqrt(5)*(x2-x1)/length)*(0.5*E4A41*(erf((x2-z_m)/sqrt(2*z_v))-erf((x1-z_m)/sqrt(2*z_v)))+\
        E4A42*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-z_m)**2/z_v)-E4A43*sqrt(0.5*z_v/pi)*exp(-0.5*(x2-z_m)**2/z_v))

    E50=1+(25*x1**2*x2**2+3*sqrt(5)*(3*length**3+5*length*x1*x2)*(x1+x2)+15*length**2*(x1**2+x2**2+3*x1*x2))/(9*length**4)
    E51=(18*sqrt(5)*length**3+15*sqrt(5)*length*(x1**2+x2**2)+(75*length**2+50*x1*x2)*(x1+x2)+60*sqrt(5)*length*x1*x2)/(9*length**4)
    E52=5*(5*x1**2+5*x2**2+15*length**2+9*sqrt(5)*length*(x1+x2)+20*x1*x2)/(9*length**4)
    E53=10*(3*sqrt(5)*length+5*x1+5*x2)/(9*length**4)
    E54=25/(9*length**4)
    muD=z_m+2*sqrt(5)*z_v/length
    E5A51=E50-muD*E51+(muD**2+z_v)*E52-(muD**3+3*z_v*muD)*E53+(muD**4+6*z_v*muD**2+3*z_v**2)*E54
    E5A52=E51-(muD+x1)*E52+(muD**2+2*z_v+x1**2+muD*x1)*E53-(muD**3+x1**3+x1*muD**2+muD*x1**2+3*z_v*x1+5*z_v*muD)*E54
    P3=exp((10*z_v-sqrt(5)*length*(x1+x2-2*z_m))/length**2)*(0.5*E5A51*(1+erf((x1-muD)/sqrt(2*z_v)))+\
        E5A52*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-muD)**2/z_v))

    jd=P1+P2+P3
    return jd