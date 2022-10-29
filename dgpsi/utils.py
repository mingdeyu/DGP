from dill import dump, load
from tabulate import tabulate
import numpy as np

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