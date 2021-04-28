import numpy as np
from imputation import imputer

class emulator:
    """Class to make predictions from the trained DGP model.

    Args:
        all_layer (list): a list that contains the trained DGP model produced by the method 'estimate'
            of the 'dgp' class. 
    """
    def __init__(self, all_layer):
        self.all_layer=all_layer
        self.n_layer=len(all_layer)
        self.imp=imputer(self.all_layer)
    
    def predict(self,x,N,method='mean_var'):
        """Implement predictions from the trained DGP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            N (int): the number of imputation to produce the predictions.
            method (str, optional): the prediction approach: mean-variance ('mean_var') or sampling 
                ('sampling') approach. Defaults to 'mean_var'.

        Returns:
            Union[ndarray, list]: if the argument method='mean_var', a list is produced and the list contains
                two numpy 2d-arrays, one for the predictive means and another for the predictive variances.
                Each array has its rows corresponding to the DGP output dimensions and columns corresponding to
                the testing positions. For example, when there are K GPs in the final layer the two arrays have
                K rows; if the argument method='sampling', a numpy 3d-array is produced. The array has the shape
                (K,N,M), where K is the DGP output dimension, N is the number of imputations, and M is the number
                of testing positions.
        """
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