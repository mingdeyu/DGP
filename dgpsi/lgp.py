import copy

import numpy as np
from tqdm import trange

__all__ = ["lgp"]


class lgp:
    """
    Class that contains the linked GP hierarchy for emulator integrations.

    Args:
        X (list): a list of L (i.e., the number of layers) sub-lists of numpy 2d-arrays of GP local input training data. 
            Each sub-list represents a layer and each numpy 2d-array in the sub-list provides the local input training 
            data produced by GPs in the last layer, except for the first sub-list that gives the global input to the GPs
            in the first layer. The array has its rows being input data points and columns being input dimensions.
        Y (list): a list of L sub-lists of numpy 2d-arrays of GP output training data. Each sub-list represents 
            a layer and each numpy 2d-array (with only one column) in the sub-list provides the output training 
            data for the corresponding GP. The array has its rows being output data points.
        all_layer (list): a list contains L (the number of layers) sub-lists, each of which contains 
            the GPs defined by the kernel class in that layer. The sub-lists are placed in the list 
            in the same order of the specified linked GP structure. 
        Z (list): a list of sub-lists of numpy 2d-arrays and Nones. Each sub-list represents a layer and the first sub-list
            should typically be set to a number of Nones because GPs in the first layer have no feeding GPs. Each numpy 2d-array 
            in the sub-list provides the additional input dimensions (i.e., those not from the feeding GPs). If a GP has no 
            external input, then set the corresponding cell in Z to None. The array has its rows being input data points and 
            columns being input dimensions. Defaults to None, meaning that there are no external inputs to any GPs.
    """

    def __init__(self, X, Y, all_layer, Z=None):
        self.X = X
        self.Y = Y
        self.all_layer = all_layer
        self.Z = Z
        self.n_layer = len(all_layer)
        self.initialize()

    def initialize(self):
        """Assign input/output data to all_layer attribute for training.
        """
        for l in range(self.n_layer):
            layer = self.all_layer[l]
            num_kernel = len(layer)
            for k in range(num_kernel):
                kernel = layer[k]
                kernel.input = copy.deepcopy(self.X[l][k])
                if self.Z != None:
                    kernel.global_input = copy.deepcopy(self.Z[l][k])
                kernel.output = copy.deepcopy(self.Y[l][k])

        # Allow method chaining
        return self

    def train(self, disable: bool = False) -> None:
        """Train the linked GP model.

        Args:
            disable (bool, optional): whether to disable the training progress bar. 
                Defaults to False.
        """
        pgb = trange(1, self.n_layer + 1, disable=disable)
        for l in pgb:
            i = 1
            for kernel in self.all_layer[l - 1]:
                kernel.maximise()
                pgb.set_description('Layer %i: Node %i' % (l, i))
                i += 1

    def predict(self, x: np.ndarray, z=None, full_layer: bool = False):
        """Implement predictions from the trained linked GP model.

        Args:
            x (ndarray): a numpy 2d-array where each row is an input testing data point and 
                each column is an input dimension.
            z (list): a list of sub-lists of numpy 2d-arrays and Nones. Each sub-list represents 
                a layer and the first sub-list should typically be set to a number of Nones because 
                GPs in the first layer have no feeding GPs. Each numpy 2d-array in the sub-list provides 
                the testing positions of additional input dimensions (i.e., those not from the feeding GPs)
                of the corresponding GP. If a GP has no external input, then set the corresponding cell and
                sub-list in z to None. The numpy 2d-array has its rows being input testing positions and 
                columns being input dimensions. Defaults to None, meaning that there are no external (testing) 
                inputs to all GPs in the linked GP structure.
            full_layer (bool, optional): whether to output the predictions of all layers. Defaults to False.

        Returns:
            tuple: a tuple is returned:
                    1. If full_layer=False, the tuple contains two numpy 2d-arrays, one for the predictive means 
                        and another for the predictive variances. Each array has its rows corresponding to testing 
                        positions and columns corresponding to DGP output dimensions (i.e., GP nodes in the final 
                        layer);
                    2. If full_layer=True, the tuple contains two lists, one for the predictive means 
                        and another for the predictive variances. Each list contains L (i.e., the number of layers) 
                        numpy 2d-arrays. Each array has its rows corresponding to testing positions and columns 
                        corresponding to output dimensions (i.e., GP nodes from the associated layer).
        """
        M = len(x)
        overall_global_test_input = x
        if full_layer:
            mu = []
            sigma2 = []
        for l in range(self.n_layer):
            layer = self.all_layer[l]
            n_kerenl = len(layer)
            overall_test_output_mean = np.empty((M, n_kerenl))
            overall_test_output_var = np.empty((M, n_kerenl))
            if l == 0:
                for k in range(n_kerenl):
                    kernel = layer[k]
                    if z is None:
                        m_k, v_k = kernel.gp_prediction(x=overall_global_test_input[:, kernel.input_dim], z=None)
                    else:
                        m_k, v_k = kernel.gp_prediction(x=overall_global_test_input[:, kernel.input_dim], z=z[l][k])
                    overall_test_output_mean[:, k], overall_test_output_var[:, k] = m_k, v_k
            else:
                for k in range(n_kerenl):
                    kernel = layer[k]
                    m_k_in, v_k_in = overall_test_input_mean[:, kernel.input_dim], overall_test_input_var[:,
                                                                                   kernel.input_dim]
                    if z is None:
                        m_k, v_k = kernel.linkgp_prediction(m=m_k_in, v=v_k_in, z=None)
                    else:
                        z_k_in = z[l][k]
                        m_k, v_k = kernel.linkgp_prediction(m=m_k_in, v=v_k_in, z=z_k_in)
                    overall_test_output_mean[:, k], overall_test_output_var[:, k] = m_k, v_k
            overall_test_input_mean, overall_test_input_var = overall_test_output_mean, overall_test_output_var
            if full_layer:
                mu.append(overall_test_input_mean)
                sigma2.append(overall_test_input_var)
        if not full_layer:
            mu = overall_test_input_mean
            sigma2 = overall_test_input_var
        return mu, sigma2
