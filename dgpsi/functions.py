from numba import jit, vectorize, float64
import numpy as np
from math import erf, exp, sqrt, pi
from numpy.random import randn

######functions for imputer########
@jit(nopython=True,cache=True)
def log_likelihood_func(y,cov,scale):
    """Compute Gaussian log-likelihood function.
    """
    cov=scale*cov
    _,logdet=np.linalg.slogdet(cov)
    quad=np.sum(y*np.linalg.solve(cov,y))
    llik=-0.5*(logdet+quad)
    return llik


@jit(nopython=True,cache=True)
def fmvn(mu,cov):
    """Generate multivariate Gaussian random samples with means.
    """
    d=len(cov)
    sn=randn(d,1)
    L=np.linalg.cholesky(cov)
    samp=(L@sn).flatten()+mu
    return samp

@jit(nopython=True,cache=True)
def k_one_matrix(X,length,name):
    """Compute the correlation matrix without the nugget term.
    """
    if name=='sexp':
        X_l=X/length
        L=np.expand_dims(np.sum(X_l**2,axis=1),axis=1)
        dis2=L-2*np.dot(X_l,X_l.T)+L.T
        K=np.exp(-dis2)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        X_l=np.expand_dims((X/length).T,axis=2)
        K1=np.ones((n,n))
        K2=np.zeros((n,n))
        for i in range(d):
            dis=np.abs(X_l[i]-X_l[i].T)
            K1*=(1+sqrt(5)*dis+5/3*dis**2)
            K2+=dis
        K2=np.exp(-sqrt(5)*K2)
        K=K1*K2
    return K

@jit(nopython=True,cache=True)
def update_f(f,nu,theta,mean):
    """Update ESS proposal samples.
    """
    fp=(f-mean)*np.cos(theta) + (nu-mean)*np.sin(theta) + mean
    return fp

@jit(nopython=True,cache=True)
def post_het1(m,v,Gamma,y_mask):
    """Calculate the conditional posterior mean and covariance of the mean 
       of the heteroskedastic Gaussian likelihood when there are no repetitions
       in the training data.
    """
    mu=m+np.sum(v*np.linalg.solve(Gamma+v,y_mask-m),axis=1)
    cov=v@np.linalg.solve(Gamma+v,Gamma)
    return mu, cov

@jit(nopython=True,cache=True)
def post_het2(m,v,Gamma,v_mask,V_mask,m_mask,y_mask):
    """Calculate the conditional posterior mean and covariance of the mean 
       of the heteroskedastic Gaussian likelihood when there are repetitions
       in the training data.
    """
    mu=m+np.sum(v_mask.T*np.linalg.solve(Gamma+V_mask,y_mask-m_mask),axis=1)
    cov=v-v_mask.T@np.linalg.solve(Gamma+V_mask,v_mask)
    return mu, cov    

######Multivariate Gaussian sampling######
@jit(nopython=True,cache=True)
def cmvn(local_input,global_input,output,scale,length,nugget,name,missingness):
    """Sample from conditional multivariate normal distribution
    """
    x=local_input[missingness,]
    w1=local_input[~missingness,]
    if global_input!=None:
        z=global_input[missingness,]
        x=np.concatenate((x, z),1)
        global_w1=global_input[~missingness,]
        w1=np.concatenate((w1, global_w1),1)
    w2=output[~missingness,]
    R=k_one_matrix(w1,length,name)+nugget*np.identity(len(w1))
    c=k_one_matrix(x,length,name)+nugget*np.identity(len(x))
    r=k_one_vec(w1,x,length,name)
    Rinv_r=np.linalg.solve(R,r)
    r_Rinv_r=r.T@Rinv_r
    v=np.abs(scale*(c-r_Rinv_r))
    m=np.sum(w2*Rinv_r,axis=0) 
    return m, v

######functions for predictions########
@jit(nopython=True,cache=True)
def gp(x,z,w1,global_w1,w2,scale,length,nugget,name):
    """Make GP predictions.
    """
    if z!=None:
        x=np.concatenate((x, z),1)
        w1=np.concatenate((w1, global_w1),1)
    R=k_one_matrix(w1,length,name)+nugget*np.identity(len(w1))
    r=k_one_vec(w1,x,length,name)
    Rinv_r=np.linalg.solve(R,r)
    r_Rinv_r=np.sum(r*Rinv_r,axis=0)
    v=np.abs(scale*(1+nugget-r_Rinv_r))
    m=np.sum(w2*Rinv_r,axis=0) 
    return m, v

@jit(nopython=True,cache=True)
def link_gp(m,v,z,w1,global_w1,w2,scale,length,nugget,name):
    """Make linked GP predictions.
    """
    M=len(m)
    m_new=np.empty((M,1))
    v_new=np.empty((M,1))
    if z!=None:
        Dw=np.shape(w1)[1]
        Dz=np.shape(z)[1]
        if len(length)==1:
            length=length*np.ones(Dw+Dz)
        w=np.concatenate((w1, global_w1),1)
        R=k_one_matrix(w,length,name)+nugget*np.identity(len(w))
        Rinv_y=np.linalg.solve(R,w2)
    else:
        Dw=np.shape(w1)[1]
        if len(length)==1:
            length=length*np.ones(Dw)
        R=k_one_matrix(w1,length,name)+nugget*np.identity(len(w1))
        Rinv_y=np.linalg.solve(R,w2)
    for i in range(M):
        mi=m[i,:]
        vi=v[i,:]
        if z!=None:
            zi=np.expand_dims(z[i,:],axis=0)
            I,J=IJ(w1,mi,vi,length[:-Dz],name)
            Iz=k_one_vec(global_w1,zi,length[-Dz::],name)
            Jz=np.dot(Iz,Iz.T)
            I,J=I*Iz,J*Jz
        else:
            I,J=IJ(w1,mi,vi,length,name)
        tr_RinvJ=np.trace(np.linalg.solve(R,J))
        IRinv_y=np.sum(I*Rinv_y)
        m_new[i]=IRinv_y
        v_new[i]=np.abs(np.sum(np.dot(Rinv_y.T,J)*Rinv_y.T)-IRinv_y**2+scale*(1+nugget-tr_RinvJ))
    return m_new.flatten(),v_new.flatten()

@jit(nopython=True,cache=True)
def k_one_vec(X,z,length,name):
    """Compute cross-correlation matrix between the testing and training input data.
    """
    if name=='sexp':
        X_l=X/length
        z_l=z/length
        L_X=np.expand_dims(np.sum(X_l**2,axis=1),axis=1)
        L_z=np.expand_dims(np.sum(z_l**2,axis=1,),axis=1)
        dis2=L_X-2*np.dot(X_l,z_l.T)+L_z.T
        k=np.exp(-dis2)
    elif name=='matern2.5':
        n=np.shape(X)[0]
        d=np.shape(X)[1]
        m=len(z)
        X_l=np.expand_dims((X/length).T,axis=2)
        z_l=np.expand_dims((z/length).T,axis=2)
        k1=np.ones((n,m))
        k2=np.zeros((n,m))
        for i in range(d):
            dis=np.abs(X_l[i]-z_l[i].T)
            k1*=(1+sqrt(5)*dis+5/3*dis**2)
            k2+=dis
        k2=np.exp(-sqrt(5)*k2)
        k=k1*k2
    return k

@jit(nopython=True,cache=True)
def IJ(X,z_m,z_v,length,name):
    """Compute I and J involved in linked GP predictions.
    """
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
            cross_L_X_z=np.dot(X_z[i],X_z[i].T)
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
        J=np.ones((n,n))
        for i in range(d):    
            if z_v[i]!=0:
                I*=np.exp((5*z_v[i]-2*sqrt(5)*length[i]*zX[i])/(2*length[i]**2))* \
                    ((1+sqrt(5)*muA[i]/length[i]+5*(muA[i]**2+z_v[i])/(3*length[i]**2))*pnorm(muA[i]/sqrt(z_v[i]))+ \
                    (sqrt(5)+(5*muA[i])/(3*length[i]))*sqrt(0.5*z_v[i]/pi)/length[i]*np.exp(-0.5*muA[i]**2/z_v[i]))+ \
                   np.exp((5*z_v[i]+2*sqrt(5)*length[i]*zX[i])/(2*length[i]**2))* \
                    ((1-sqrt(5)*muB[i]/length[i]+5*(muB[i]**2+z_v[i])/(3*length[i]**2))*pnorm(-muB[i]/sqrt(z_v[i]))+ \
                    (sqrt(5)-(5*muB[i])/(3*length[i]))*sqrt(0.5*z_v[i]/pi)/length[i]*np.exp(-0.5*muB[i]**2/z_v[i]))
                for k in range(n):
                    for l in range(k+1):
                        J_lk=Jd(X[i][l][0],X[i][k][0],z_m[i],z_v[i],length[i])
                        if l==k:
                           J[l,k]*=J_lk
                        else:
                           J[l,k]*=J_lk
                           J[k,l]*=J_lk
            else:
                Id=(1+sqrt(5)*np.abs(zX[i])/length[i]+5*zX[i]**2/(3*length[i]**2))*np.exp(-sqrt(5)*np.abs(zX[i])/length[i])
                I*=Id
                J*=np.dot(Id,Id.T)
    return I,J

@vectorize([float64(float64)],nopython=True,cache=True)
def pnorm(x):
    """Compute standard normal CDF.
    """
    return 0.5*(1+erf(x/sqrt(2)))    

@jit(nopython=True,cache=True,fastmath=True)
def Jd(X1,X2,z_m,z_v,length):
    """Compute J components in different input dimensions for Matern2.5 kernel.
    """
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