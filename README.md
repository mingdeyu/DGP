# dgpsi
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/mingdeyu/DGP?display_name=release&include_prereleases&style=flat-square)](https://github.com/mingdeyu/DGP/releases)
[![Read the Docs (version)](https://img.shields.io/readthedocs/dgpsi/latest?style=flat-square)](https://dgpsi.readthedocs.io)
[![Conda](https://img.shields.io/conda/dn/conda-forge/dgpsi?label=Conda%20Downloads&style=flat-square)](https://anaconda.org/conda-forge/dgpsi)
![Conda](https://img.shields.io/conda/pn/conda-forge/dgpsi?color=orange&style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.1137%2F20M1323771-informational?style=flat-square)](https://epubs.siam.org/doi/abs/10.1137/20M1323771)

## A Python package for deep and linked Gaussian process emulations
`dgpsi` currently implements:

* Deep Gaussian process emulation with flexible architecture construction: 
    - multiple layers;
    - multiple GP nodes;
    - separable or non-separable squared exponential and Mat&eacute;rn2.5 kernels;
    - global input connections;
    - non-Gaussian likelihoods (Poisson, Negative-Binomial, heteroskedastic Gaussian, and more to come);
* Linked emulation of feed-forward systems of computer models:
    - linking GP emulators of deterministic individual computer models;
    - linking GP and DGP emulators of deterministic individual computer models;
* **(New Feature)** Multi-core predictions from GP, DGP, and Linked (D)GP emulators.

## Installation
`dgpsi` currently requires Python version 3.7, 3.8, or 3.9. The package can be installed via `pip`:

```bash
pip install dgpsi
```

or `conda`:

```bash
conda install -c conda-forge dgpsi
```

However, to gain the best performance of the package or you are using an Apple Silicon computer, we recommend the following steps for the installation:
* Download and install `Miniforge3` that is compatible to your system from [here](https://github.com/conda-forge/miniforge).
* Run the following command in your terminal app to create a virtual environment called `dgp_si`:

```bash
conda create -n dgp_si python=3.9.13 
```

* Activate and enter the virtual environment:

```bash
conda activate dgp_si
```

* Install `dgpsi`:
    - for Apple Silicon users, you could gain speed-up by switching to Apple's Accelerate framework:

    ```bash
    conda install dgpsi "libblas=*=*accelerate"
    ```

    - for Intel users, you could gain speed-up by switching to MKL:

    ```bash
    conda install dgpsi "libblas=*=*mkl"
    ```

    - otherwise, simply run:
    ```bash
    conda install dgpsi
    ```

## Demo and Documentation
Please see [demo](https://github.com/mingdeyu/DGP/tree/master/demo) for some illustrative examples of the method. The API reference 
of the package can be accessed from [https://dgpsi.readthedocs.io](https://dgpsi.readthedocs.io), and some tutorials will be soon added there.

## Tips
* Since SI is a stochastic inference, in case of unsatisfactory results, you may want to try to restart the training multiple times even with initial values of hyperparameters unchanged;
* The recommended DGP structure is a two-layered one with the number of GP nodes in the first layer equal to the number of input dimensions (i.e., number of input columns) and the number of GP nodes in the second layer equal to the number of output dimensions (i.e., number of output columns) or the number of parameters in the specified likelihood. The `dgp` class in the package is default to this structure.

## Contact
Please feel free to email me with any questions and feedbacks: 

Deyu Ming <[deyu.ming.16@ucl.ac.uk](mailto:deyu.ming.16@ucl.ac.uk)>.

## References
> [Ming, D., Williamson, D., and Guillas, S. (2021) Deep Gaussian process emulation using stochastic imputation.](https://arxiv.org/abs/2107.01590)

> [Ming, D. and Guillas, S. (2021) Linked Gaussian process emulation for systems of computer models using Mat&eacute;rn kernels and adaptive design, <i>SIAM/ASA Journal on Uncertainty Quantification</i>. 9(4), 1615-1642.](https://epubs.siam.org/doi/abs/10.1137/20M1323771)
