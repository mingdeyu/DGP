import numpy as np
from elliptical_slice import imputer

class emulator:
    def __init__(self, all_layer):
        self.all_layer=all_layer
        self.n_layer=len(all_layer)
        self.imp=imputer(self.all_layer)
    
    def predict(self,x,N,method='mean_var'):
        #warm up
        M=len(x)
        (self.imp).sample(burnin=50)
        #start predictions
        mean_pred=[]
        variance_pred=[]
        for _ in range(N):
            overall_global_test_input=x
            (self.imp).sample()
            for l in range(self.n_layer):
                layer=self.all_layer[l]
                n_kerenl=len(layer)
                overall_test_output_mean=np.empty((M,n_kerenl))
                overall_test_output_var=np.empty((M,n_kerenl))
                if l==0:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k,v_k=kernel.gp_prediction(x=overall_global_test_input[:,kernel.input_dim],z=None)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                else:
                    for k in range(n_kerenl):
                        kernel=layer[k]
                        m_k_in,v_k_in=overall_test_input_mean[:,kernel.input_dim],overall_test_input_var[:,kernel.input_dim]
                        if np.any(kernel.connect!=None):
                            z_k_in=overall_global_test_input[:,kernel.connect]
                        else:
                            z_k_in=None
                        m_k,v_k=kernel.linkgp_prediction(m=m_k_in,v=v_k_in,z=z_k_in)
                        overall_test_output_mean[:,k],overall_test_output_var[:,k]=m_k,v_k
                overall_test_input_mean,overall_test_input_var=overall_test_output_mean,overall_test_output_var
            mean_pred.append(overall_test_input_mean)
            variance_pred.append(overall_test_input_var)
        if method=='sampling':
            samples=[]
            for mu, sigma2 in zip(mean_pred, variance_pred):
                realisation=np.random.normal(mu,np.sqrt(sigma2))
                samples.append(realisation)
            samples=np.asarray(samples)
            return samples.transpose(2,0,1)
        elif method=='mean_var':
            mean_pred=np.asarray(mean_pred)
            variance_pred=np.asarray(variance_pred)
            mu=np.mean(mean_pred,axis=0)
            sigma2=np.mean((mean_pred**2+variance_pred),axis=0)-mu**2
            return mu.T, sigma2.T