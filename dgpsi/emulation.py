import numpy as np
from .imputation import imputer

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
    
    def predict(self,x,N=50,method='mean_var',full_layer=False):
        """Implement predictions from the trained DGP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            N (int): the number of imputation to produce the predictions.
            method (str, optional): the prediction approach: mean-variance ('mean_var') or sampling 
                ('sampling') approach. Defaults to 'mean_var'.
            full_layer (bool, optional): whether to output the predictions of all layers. Defaults to False.

        Returns:
            Union[tuple, list]: if the argument method='mean_var', a tuple is returned:
                    1. If full_layer=False, the tuple contains two numpy 2d-arrays, one for the predictive means 
                        and another for the predictive variances. Each array has its rows corresponding to testing 
                        positions and columns corresponding to DGP output dimensions (i.e., GP nodes in the final layer);
                    2. If full_layer=True, the tuple contains two lists, one for the predictive means 
                        and another for the predictive variances. Each list contains L (i.e., the number of layers) 
                        numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns 
                        corresponding to output dimensions (i.e., GP nodes from the associated layer).
                if the argument method='sampling', a list is returned:
                    1. If full_layer=False, the list contains D (i.e., the number of GP nodes in the final layer) numpy 
                        2d-arrays. Each array has its rows corresponding to testing positions and columns corresponding to
                        N imputations;
                    2. If full_layer=True, the list contains L (i.e., the number of layers) sub-lists. Each sub-list 
                        represents the samples draw from the GPs in the corresponding layers, and contains D (i.e., the 
                        number of GP nodes in the corresponding layer) numpy 2d-arrays. Each array gives samples of the
                        output from one of D GPs at the testing positions, and has its rows corresponding to testing 
                        positions and columns corresponding to N imputations.
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
            if full_layer==True:
                mean_pred_oneN=[]
                variance_pred_oneN=[]
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
                if full_layer==True:
                    mean_pred_oneN.append(overall_test_input_mean)
                    variance_pred_oneN.append(overall_test_input_var)
            if full_layer==True:
                mean_pred.append(mean_pred_oneN)
                variance_pred.append(variance_pred_oneN)
            else:
                mean_pred.append(overall_test_input_mean)
                variance_pred.append(overall_test_input_var)
        if method=='sampling':
            if full_layer==True:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                samples=[]
                for l in range(self.n_layer):
                    samples_layerwise=[]
                    for mu, sigma2 in zip(mu_layerwise[l], var_layerwise[l]):
                        realisation=np.random.normal(mu,np.sqrt(sigma2))
                        samples_layerwise.append(realisation)
                    samples_layerwise=np.asarray(samples_layerwise).transpose(2,1,0)
                    samples.append(list(samples_layerwise))
            else:
                samples=[]
                for mu, sigma2 in zip(mean_pred, variance_pred):
                    realisation=np.random.normal(mu,np.sqrt(sigma2))
                    samples.append(realisation)
                samples=list(np.asarray(samples).transpose(2,1,0))
            return samples
        elif method=='mean_var':
            if full_layer==True:
                mu_layerwise=[list(mean_n) for mean_n in zip(*mean_pred)]
                var_layerwise=[list(var_n) for var_n in zip(*variance_pred)]
                mu=[np.mean(mu_l,axis=0) for mu_l in mu_layerwise]
                mu2_mean=[np.mean(np.square(mu_l),axis=0) for mu_l in mu_layerwise]
                var_mean=[np.mean(var_l,axis=0) for var_l in var_layerwise]
                sigma2=[i+j-k**2 for i,j,k in zip(mu2_mean,var_mean,mu)]
            else:
                mu=np.mean(mean_pred,axis=0)
                sigma2=np.mean((np.square(mean_pred)+variance_pred),axis=0)-mu**2
            return mu, sigma2