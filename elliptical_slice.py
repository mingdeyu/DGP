from numpy.random import uniform
import numpy as np
from functions import log_likelihood_func, mvn, k_one_matrix, update_f

class ess:
    def __init__(self, all_kernel,X,Y,ini):
        self.all_kernel=all_kernel
        self.X=X
        self.Y=Y
        self.ini=ini
        self.sample = []

    def sample_ess(self,N,burnin):
        n_layer=len(self.all_kernel)
        if not self.sample:
            total_samples = N + burnin
            sample=[np.tile(self.X,(total_samples,1,1))]
            for i in range(n_layer-1):
                samp=np.zeros((total_samples,len(self.Y),1))
                samp[0]=self.ini[i]
                sample.append(samp)
            sample.append(np.tile(self.Y,(total_samples,1,1)))

            for t in range(1, total_samples):
                for i in range(n_layer-1):
                    if i==0:
                        sample[i+1][t]=self.one_sample(x=sample[i][t-1],y=sample[i+2][t-1],f=sample[i+1][t-1],k1=self.all_kernel[i],k2=self.all_kernel[i+1])
                    else:
                        sample[i+1][t]=self.one_sample(x=sample[i][t],y=sample[i+2][t-1],f=sample[i+1][t-1],k1=self.all_kernel[i],k2=self.all_kernel[i+1])
        else:
            t0=len(self.sample[0])
            add_samples = N
            extra_sample=[np.tile(self.X,(add_samples,1,1))]
            for i in range(n_layer-1):
                extra_samp=np.zeros((add_samples,len(self.Y),1))
                extra_sample.append(extra_samp)
            extra_sample.append(np.tile(self.Y,(add_samples,1,1)))
            sample=[np.concatenate((i,j),axis=0) for i,j in zip(self.sample,extra_sample)]
            
            for t in range(t0, t0+add_samples):
                for i in range(n_layer-1):
                    if i==0:
                        sample[i+1][t]=self.one_sample(x=sample[i][t-1],y=sample[i+2][t-1],f=sample[i+1][t-1],k1=self.all_kernel[i],k2=self.all_kernel[i+1])
                    else:
                        sample[i+1][t]=self.one_sample(x=sample[i][t],y=sample[i+2][t-1],f=sample[i+1][t-1],k1=self.all_kernel[i],k2=self.all_kernel[i+1])
        self.sample=sample
        return [t[burnin:] for t in self.sample]

    def one_sample(self,x,y,f,k1,k2):
        mean=np.zeros(len(y))
        f=f.flatten()
        y=y.flatten()
        
        covariance=k_one_matrix(x,k1.length,k1.nugget,k1.name)
        # Choose the ellipse for this sampling iteration.
        #nu = multivariate_normal(np.zeros(mean.shape), covariance)
        nu = mvn(covariance,k1.scale,k1.mean_prior,k1.zero_mean)
        # Set the candidate acceptance threshold.
        cov_f=k_one_matrix(f.reshape([-1,1]),k2.length,k2.nugget,k2.name)
        log_y = log_likelihood_func(y,cov_f,k2.scale,k2.mean_prior,k2.zero_mean) + np.log(uniform())
        # Set the bracket for selecting candidates on the ellipse.
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Iterates until an candidate is selected.
        while True:
            # Generates a point on the ellipse defines by `nu` and the input. We
            # also compute the log-likelihood of the candidate and compare to
            # our threshold.
            fp = update_f(f,mean,nu,theta)
            cov_fp=k_one_matrix(fp.reshape([-1,1]),k2.length,k2.nugget,k2.name)
            log_fp = log_likelihood_func(y,cov_fp,k2.scale,k2.mean_prior,k2.zero_mean)
            if log_fp > log_y:
                return fp.reshape([-1,1])
            else:
                # If the candidate is not selected, shrink the bracket and
                # generate a new `theta`, which will yield a new candidate
                # point on the ellipse.
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = uniform(theta_min, theta_max)
