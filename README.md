# dgpsi
`dgpsi` implements inference of both deep and linked Gaussian process emulation using stochastic imputation. 

## Key features
`dgpsi` currently has the following features:

* Deep Gaussian process emulation with flexible architecture construction: 
    - multiple layers;
    - multiple GP nodes;
    - separable or non-separable squared exponential and Mat&eacute;rn2.5 kernels;
    - global input connections;
    - non-Gaussian likelihoods (Poisson, Negative-Binomial, heteroskedastic Gaussian, and more to come);
* Linked emulation of feed-forward systems of computer models:
    - linking GP emulators of deterministic individual computer models;
    - linking GP and DGP emulators of deterministic individual computer models;
* **(New Feature)** Multi-core predictions from GP, DGP, and Linked (D)GP emulators;
* More features coming soon.

Please see [demo](https://github.com/mingdeyu/DGP/tree/master/demo) for some illustrative examples of the method. The API reference 
of the package can be accessed from [https://dgpsi.readthedocs.io](https://dgpsi.readthedocs.io).

## Installation
The simplest way to install the package is to use `pip`:

```bash
pip install dgpsi
```

However, to gain the best performance of the package, we recommend the following steps for the installation:
* Download and install `Miniforge3` that is compatible to your system from [here](https://github.com/conda-forge/miniforge).
* Run the following command in your terminal app to create a virtual environment called `dgp_si`:

```bash
conda create -n dgp_si python=3.9.13 
```

* Activate and enter the virtual environment:

```bash
conda activate dgp_si
```

* Install required packages:
    - for Mac Silicon chip users, you could gain speed-up by switching to Apple's Accelerate framework by running:

    ```bash
    conda install -c conda-forge cython pybind11 matplotlib tqdm jupyter dill pathos psutil numpy pythran scipy scikit-learn scikit-build numba "libblas=*=*accelerate"
    ```

    - for Intel chip users, you could gain speed-up by switching to MKL by running:

    ```bash
    conda install -c conda-forge cython pybind11 matplotlib tqdm jupyter dill pathos psutil numpy pythran scipy scikit-learn scikit-build numba "libblas=*=*mkl"
    ```

    - otherwise, simply run:
    ```bash
    conda install -c conda-forge cython pybind11 matplotlib tqdm jupyter dill pathos psutil numpy pythran scipy scikit-learn scikit-build numba
    ```

* Install `dgpsi` package:

```bash
cd A-LOCATION-YOU-PREFER
git clone https://github.com/mingdeyu/DGP.git
cd DGP
pip install .
```

## Tips
* Since SI is a stochastic inference, in case of unsatisfactory results, you may want to try to restart the training multiple times even with initial values of hyperparameters unchanged;
* The recommended DGP structure is a two-layered one with the number of GP nodes in the first layer equal to the number of input dimensions (i.e., number of input columns) and the number of GP nodes in the second layer equal to the number of output dimensions (i.e., number of output columns) or the number of parameters in the specified likelihood.

## Contact
Please feel free to email me with any questions and feedbacks: 

Deyu Ming <[deyu.ming.16@ucl.ac.uk](mailto:deyu.ming.16@ucl.ac.uk)>.

## References
> [Ming, D., Williamson, D., and Guillas, S. (2021) Deep Gaussian process emulation using stochastic imputation.](https://arxiv.org/abs/2107.01590)

> [Ming, D. and Guillas, S. (2021) Linked Gaussian process emulation for systems of computer models using Mat&eacute;rn kernels and adaptive design, <i>SIAM/ASA Journal on Uncertainty Quantification</i>. 9(4), 1615-1642.](https://epubs.siam.org/doi/abs/10.1137/20M1323771)
