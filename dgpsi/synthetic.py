import copy
import numpy as np
from .functions import k_one_matrix

__all__ = ["path"]


class path:
    # main algorithm
    def __init__(self, X, all_layer):
        self.X = X
        self.n_layer = len(all_layer)
        self.all_layer = copy.deepcopy(all_layer)
        for l in range(self.n_layer):
            layer = self.all_layer[l]
            num_kernel = len(layer)
            for k in range(num_kernel):
                kernel = layer[k]
                if np.any(kernel.connect is not None):
                    kernel.global_input = copy.deepcopy(self.X[:, kernel.connect])

    def generate(self, N):
        d = len(self.all_layer[-1])
        m = len(self.X)
        path = np.empty((N, m, d))
        for i in range(N):
            x = self.X
            for l in range(self.n_layer):
                layer = self.all_layer[l]
                num_kernel = len(layer)
                out = np.empty((m, num_kernel))
                for k in range(num_kernel):
                    kernel = layer[k]
                    if np.any(kernel.input_dim is not None):
                        In = x[:, kernel.input_dim]
                    else:
                        In = x
                    if np.any(kernel.connect is not None):
                        In = np.concatenate((In, kernel.global_input), 1)
                    cov = (k_one_matrix(In, kernel.length, kernel.name) + kernel.nugget * np.identity(m)) * kernel.scale
                    L = np.linalg.cholesky(cov)
                    randn = np.random.normal(size=[m, 1])
                    out[:, k] = (L @ randn).flatten()
                x = out
            path[i,] = x
        return path.transpose(2, 0, 1)
