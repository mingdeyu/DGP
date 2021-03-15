import numpy as np

class path:
    #main algorithm
    def __init__(self, X, all_kernel):
        self.X=X
        self.layer=len(all_kernel)
        self.all_kernel=all_kernel
        for kernel in self.all_kernel:
            if kernel.connect==1:
                kernel.global_input=self.X

    def generate(self,N):
        n=len(self.X)
        path=np.empty((N,n))
        for i in range(N):
            x=self.X
            for kernel in self.all_kernel:
                cov=kernel.k_matrix(x)*kernel.scale
                L=np.linalg.cholesky(cov)
                randn=np.random.normal(size=[n,1])
                x=L@randn
            path[i,]=x.flatten()  
        return path.T