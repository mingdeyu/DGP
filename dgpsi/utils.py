from dill import dump, load
from tabulate import tabulate
from numba import njit, set_num_threads, get_num_threads
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize, Bounds
from sklearn.metrics.pairwise import pairwise_kernels
from pathos.multiprocessing import ProcessingPool as Pool
from numba import set_num_threads
from typing import Callable, Tuple, Any, Optional
import multiprocess.context as ctx
import platform
import psutil 
#import copy
#from contextlib import contextmanager

######Save and Load Emulators#######
def write(emu, pkl_file):
    """Save the constructed emulator to a `.pkl` file.
    
    Args:
        emu (class): an emulator class. For GP, it is the :class:`.gp` class after training. 
            For DGP, it is the :class:`.emulator` class. For linked GP/DGP, it is the :class:`.lgp` class.
        pkl_file (strings): the path to and the name of the `.pkl` file to which
            the emulator specified by **emu** is saved.
    """
    dump(emu, open(pkl_file+".pkl","wb"))


def read(pkl_file):
    """Load the `.pkl` file that stores the emulator.
    
    Args:
        pkl_file (strings): the path to and the name of the `.pkl` file where
            the emulator is stored.
    
    Returns:
        class: an emulator class. For GP, it is the :class:`.gp` class. For DGP, it is the :class:`.emulator` class. 
        For linked GP/DGP, it is the :class:`.lgp` class.
    """
    emu = load(open(pkl_file+".pkl", "rb"))
    return emu

#@contextmanager
#def modify_all_layer_set(instance):
#    original_all_layer_set = copy.deepcopy(instance.all_layer_set)
#    yield instance.all_layer_set
#    instance.all_layer_set = original_all_layer_set

######seed function#######
@njit(cache=True)
def nb_seed(value):
    """Set seed for Numba functions.
    """
    np.random.seed(value)

######thread number function#######
def get_thread():
    """Get number of numba thread.
    """
    return get_num_threads()

def set_thread(value):
    """Set number of numba thread.
    """
    set_num_threads(value)

######summary function#######
def summary(obj, tablefmt='fancy_grid'):
    """Summarize key information of GP, DGP, and Linked (D)GP structures.

    Args:
        obj (class): **obj** can be one of the following:

            1. an instance of :class:`.kernel` class;
            2. an instance of :class:`.gp` class;
            3. an instance of :class:`.dgp` class;
            4. an instance of :class:`.emulator` class;
            5. an instance of :class:`.lgp` class
        tablefmt (str): the style of output summary table. See https://pypi.org/project/tabulate/ for different options.
            Defaults to `fancy_grid`.
    
    Returns:
        string: a table summarizing key information contained in **obj**.
    """
    info=[]
    if type(obj).__name__=='kernel':
        info.append(['Kernel Fun', 'Length-scale(s)', 'Variance', 'Nugget'])
        info.append(['Squared-Exp' if obj.name=='sexp' else 'Matern-2.5',
        f"{np.array2string(obj.length, precision=3, floatmode='fixed', separator=', ')}", 
        f"{np.array2string(np.atleast_1d(obj.scale)[0], precision=3, floatmode='fixed')}" if obj.scale_est else f"{np.array2string(np.atleast_1d(obj.scale)[0], precision=3, floatmode='fixed')} (fixed)", 
        f"{np.array2string(np.atleast_1d(obj.nugget)[0], precision=3, floatmode='fixed')}" if obj.nugget_est else f"{np.array2string(np.atleast_1d(obj.nugget)[0], precision=3, floatmode='fixed')} (fixed)",
        ])
        table = tabulate(info, headers='firstrow', tablefmt=tablefmt)
        print(table)
    elif type(obj).__name__=='gp':
        ker=obj.kernel
        info.append(['Kernel Fun', 'Length-scale(s)', 'Variance', 'Nugget', 'Input Dims'])
        info.append(['Squared-Exp' if ker.name=='sexp' else 'Matern-2.5',
        f"{np.array2string(ker.length, precision=3, floatmode='fixed', separator=', ')}", 
        f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')}" if ker.scale_est else f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')} (fixed)", 
        f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')}" if ker.nugget_est else f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')} (fixed)",
        f"{np.array2string(ker.input_dim+1, separator=', ')}" if ker.connect is None else f"{np.array2string(np.concatenate((ker.input_dim+1,ker.connect+1)), separator=', ')}"])
        table = tabulate(info, headers='firstrow', tablefmt=tablefmt)
        print(table)
        print("'Input Dims' indicates the dimensions (i.e., column indices) of your input data that are used for GP emulator training.")
    elif type(obj).__name__=='dgp':
        if obj.N!=0:
            print('To get the summary of the trained DGP model, construct an emulator instance using the emulator() class and then apply summary() to it.')
            return
        all_layer = obj.all_layer
        info.append(['Layer No.', 'Node No.', 'Type', 'Length-scale(s)', 'Variance', 'Nugget', 'Input Dims', 'Global Connection'])
        for l in range(obj.n_layer):
            layer=all_layer[l]
            for k in range(len(layer)):
                ker=layer[k]
                info.append([f'Layer {l+1:d}', f'Node {k+1:d}',
                    'GP (Squared-Exp)' if ker.name=='sexp' else 'GP (Matern-2.5)' if ker.name=='matern2.5' else f'Likelihood ({ker.name})',
                    'NA' if ker.type=='likelihood' else f"{np.array2string(ker.length, precision=3, floatmode='fixed', separator=', ')}", 
                    'NA' if ker.type=='likelihood' else f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')}" if ker.scale_est else f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')} (fixed)", 
                    'NA' if ker.type=='likelihood' else f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')}" if ker.nugget_est else f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')} (fixed)",
                    f"{np.array2string(ker.input_dim+1, separator=', ')}" if l!=0 else f"{np.array2string(ker.input_dim+1, separator=', ')}" if ker.connect is None else f"{np.array2string(np.concatenate((ker.input_dim+1,ker.connect+1)), separator=', ')}",
                    'NA' if ker.type=='likelihood' else 'No' if l==0 else f"{np.array2string(ker.connect+1, separator=', ')}" if ker.connect is not None else 'No'])
        table = tabulate(info, headers='firstrow', tablefmt=tablefmt)
        print(table)
        print("1. 'Input Dims' presents the indices of GP nodes in the feeding layer whose outputs feed into the GP node referred by 'Layer No.' and 'Node No.'.")
        print("2. 'Global Connection' indicates the dimensions (i.e., column indices) of the global input data that are used as additional input dimensions to the GP node referred by 'Layer No.' and 'Node No.'.")
    elif type(obj).__name__=='emulator':
        all_layer = obj.all_layer
        info.append(['Layer No.', 'Node No.', 'Type', 'Length-scale(s)', 'Variance', 'Nugget', 'Input Dims', 'Global Connection'])
        for l in range(obj.n_layer):
            layer=all_layer[l]
            for k in range(len(layer)):
                ker=layer[k]
                info.append([f'Layer {l+1:d}', f'Node {k+1:d}',
                    'GP (Squared-Exp)' if ker.name=='sexp' else 'GP (Matern-2.5)' if ker.name=='matern2.5' else f'Likelihood ({ker.name})',
                    'NA' if ker.type=='likelihood' else f"{np.array2string(ker.length, precision=3, floatmode='fixed', separator=', ')}", 
                    'NA' if ker.type=='likelihood' else f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')}" if ker.scale_est else f"{np.array2string(np.atleast_1d(ker.scale)[0], precision=3, floatmode='fixed')} (fixed)", 
                    'NA' if ker.type=='likelihood' else f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')}" if ker.nugget_est else f"{np.array2string(np.atleast_1d(ker.nugget)[0], precision=3, floatmode='fixed')} (fixed)",
                    f"{np.array2string(ker.input_dim+1, separator=', ')}" if l!=0 else f"{np.array2string(ker.input_dim+1, separator=', ')}" if ker.connect is None else f"{np.array2string(np.concatenate((ker.input_dim+1,ker.connect+1)), separator=', ')}",
                    'NA' if ker.type=='likelihood' else 'No' if l==0 else f"{np.array2string(ker.connect+1, separator=', ')}" if ker.connect is not None else 'No'])
        table = tabulate(info, headers='firstrow', tablefmt=tablefmt)
        print(table)
        print("1. 'Input Dims' presents the indices of GP nodes in the feeding layer whose outputs feed into the GP node referred by 'Layer No.' and 'Node No.'.")
        print("2. 'Global Connection' indicates the dimensions (i.e., column indices) of the global input data that are used as additional input dimensions to the GP node referred by 'Layer No.' and 'Node No.'.")
    elif type(obj).__name__=='lgp':
        all_layer = obj.all_layer
        info.append(['Layer No.', 'Emulator No.', 'Type', 'Connection', 'External Inputs'])
        for l in range(obj.L):
            layer=all_layer[l]
            for k in range(len(layer)):
                cont=layer[k]
                if l==0:
                    links = f"Global input: {np.array2string(cont.local_input_idx+1, separator=', ')}"
                    external = 'No'
                else:
                    if isinstance(cont.local_input_idx, list):
                        local_input_idx = cont.local_input_idx
                    else:
                        local_input_idx=[None]*(l-1)
                        local_input_idx.append(cont.local_input_idx)
                    connected_emu, connected_output = [], []
                    for i in range(l):
                        emu_idx, output_idx, emu_count = [], [], 0
                        for feeding_cont in all_layer[i]:
                            n = 1 if feeding_cont.type=='gp' else len(feeding_cont.structure[-1])
                            emu_idx, output_idx = np.concatenate((emu_idx, np.array([emu_count]*n))), np.concatenate((output_idx, np.arange(n)))
                            emu_count += 1
                        idx = local_input_idx[i]
                        if idx is not None:
                            connected_emu.append( emu_idx[idx] )
                            connected_output.append( output_idx[idx] )
                        else:
                            connected_emu.append( None )
                            connected_output.append( None )
                    links = ''
                    for i in range(len(local_input_idx)):
                        if local_input_idx[i] is not None:
                            for j in range(len(local_input_idx[i])):
                                links += f"Emu {np.int64(connected_emu[i][j]+1)} in Layer {i+1}: output {np.int64(connected_output[i][j]+1)}\n"
                    if cont.type == 'gp':
                        external = 'No' if cont.structure.connect is None else 'Yes'
                    else:
                        external = 'No' if cont.structure[0][0].connect is None else 'Yes'
                info.append([f'Layer {l+1:d}', f'Emu {k+1:d}', 'DGP' if cont.type=='dgp' else 'GP', links, external
                    ])
        table = tabulate(info, headers='firstrow', tablefmt=tablefmt)
        print(table)
        print("1. 'Connection' gives the indices of emulators and the associated output dimensions that are linked to the emulator referred by 'Layer No.' and 'Emulator No.'.")
        print("2. 'External Inputs' indicates if the emulator (referred by 'Layer No.' and 'Emulator No.') has external inputs that are not provided by the feeding emulators.")

def have_same_shape(list1, list2):
    if len(list1) != len(list2):
        return False
    for sublist1, sublist2 in zip(list1, list2):
        if isinstance(sublist1, list) and isinstance(sublist2, list):
            if not have_same_shape(sublist1, sublist2):
                return False
        elif isinstance(sublist1, list) or isinstance(sublist2, list):
            return False
    return True

class NystromKPCA():
    def __init__(self, n_components, m = 200):
        self.m = m 
        self.n_components = n_components
        self.basis_inds = None

    def fit_transform(self, X):
        n_samples = X.shape[0]
        self.m = min(n_samples, self.m)
        inds = np.random.permutation(n_samples)
        self.basis_inds = inds[:self.m]
        basis = X[self.basis_inds]

        K_nm = pairwise_kernels(
            X,
            basis,
            metric='sigmoid',
            filter_params=True
        )

        K_mm = K_nm[self.basis_inds]

        K_mm_p, K_nm_p = self.demean_matrices(K_mm, K_nm)

        K_inv_sqrt = self.get_inverse(K_mm_p, is_sqrt = True)

        nystrom_matrix = K_inv_sqrt @ K_nm_p.T @ K_nm_p @ K_inv_sqrt / n_samples
        _, U = np.linalg.eigh(nystrom_matrix)
        U = U[:,::-1]

        components_ = K_inv_sqrt @ U[:,:self.n_components]
        scores_     = K_nm_p @ components_

        scores_ = self.flip_dimensions(scores_)

        #self.K_mm, self.K_mm_p, self.K_nm, self.K_nm_p = K_mm, K_mm_p, K_nm, K_nm_p

        return scores_
    
    def demean_matrices(self, K_mm, K_nm):
        n, m = K_nm.shape
        n_mean = K_nm.sum(0) / n
        M1 = np.tile(n_mean, (n,1))
        m0 = self.get_inverse(K_mm) @ n_mean[:,np.newaxis]
        M2 = np.tile(K_nm @ m0, (1, m))
        M3 = n_mean @ m0
        K_nm_p = K_nm - M1 - M2 + M3
        M1 = M1[:m]
        K_mm_p = K_mm - M1 - M1.T + M3
        return K_mm_p, K_nm_p
    
    @staticmethod
    def get_inverse(K, is_sqrt = False):
        U, S, V = svd(K)
        S = np.maximum(S, 1e-12)
        if is_sqrt:
            K_inv = np.dot(U / np.sqrt(S), V)
        else:
            K_inv = np.dot(U / S, V)
        return K_inv
    
    @staticmethod
    def flip_dimensions(scores):
        flip = (scores.min(0) + scores.max(0)) / 2 < 0
        flip_matrix = np.diag(1 - 2 * flip)
        scores_flipped = scores @ flip_matrix
        return scores_flipped
    
def multistart(
    func: Callable[[np.ndarray, Any], float],
    initials: np.ndarray,
    lb: np.ndarray,
    up: np.ndarray,
    args: Tuple = (),
    method: str = 'L-BFGS-B',
    core_num: Optional[int] = None,
    out_dim: Optional[int] = 0,
    int_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Perform parallel multistart optimization and return the best optimized x.
    
    Parameters:
    - func: The objective function to be minimized. Should accept (x, *args).
    - initials: 2D NumPy array where each row is a starting point.
    - lb: 1D NumPy array of lower bounds for each parameter.
    - up: 1D NumPy array of upper bounds for each parameter.
    - args: Additional arguments to pass to the objective function.
    - method: Optimization method (default 'L-BFGS-B').
    - core_num: Number of worker processes to use.
    - out_dim: The index of the output to which the optimization is to be implemented.
    - int_mask: Boolean mask indicating which variables in `x` must be integers.
    
    Returns:
    - best_x: Optimized parameters corresponding to the lowest target value.
    """
    
    # Create a Bounds object using Scipy's Bounds class
    bounds = Bounds(lb, up)
    D = len(lb)
    os_type = platform.system()
    if os_type in ['Darwin', 'Linux']:
        ctx._force_start_method('forkserver')
    total_cores = psutil.cpu_count(logical = False)
    if core_num is None:
        core_num = total_cores//2
    num_thread = total_cores // core_num

    def wrapped_func(x, *args):
        if int_mask is not None:
            x[int_mask] = np.round(x[int_mask])
        # Convert 1D x0 to 2D array with one row
        x_2d = np.atleast_2d(x)
        # Compute negative of the function
        if out_dim == -1:
            return -np.mean(func(x_2d, *args)[0])
        else:
            return -func(x_2d, *args)[0][out_dim]
    
    def optimize_from_x0(x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function from a single starting point.
        
        Parameters:
        - x0: Initial starting point.
        
        Returns:
        - Tuple containing the optimized x and the corresponding fun value.
        """
        set_num_threads(num_thread)
        res = minimize(
            wrapped_func,
            x0,
            args=args,
            method=method,
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': np.max((30,20+5*D))}
        )
        return res.x, res.fun
    
    if core_num == 1:
        results = [optimize_from_x0(x) for x in initials]
    else:
    # Create a pool of worker processes
        with Pool(core_num) as pool:
            results = pool.map(optimize_from_x0, [x for x in initials])
    
    # Convert results to NumPy arrays for efficient processing
    optimized_x, fun_values = zip(*results)
    optimized_x = np.array(optimized_x)
    fun_values = np.array(fun_values)
    
    # Identify the index of the minimum fun value
    best_idx = np.argmin(fun_values)
    
    # Extract the best optimized x
    best_x = optimized_x[best_idx]

    if int_mask is not None:
        best_x[int_mask] = np.round(best_x[int_mask])
    
    return best_x