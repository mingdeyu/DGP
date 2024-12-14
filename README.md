# dgpsi
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/mingdeyu/DGP?display_name=release&include_prereleases&style=flat-square)](https://github.com/mingdeyu/DGP/releases)
[![Conda](https://img.shields.io/conda/dn/conda-forge/dgpsi?label=Conda%20Downloads&style=flat-square)](https://anaconda.org/conda-forge/dgpsi)
![Conda](https://img.shields.io/conda/pn/conda-forge/dgpsi?color=orange&style=flat-square)
[![Read the Docs (version)](https://img.shields.io/readthedocs/dgpsi/latest?style=flat-square)](https://dgpsi.readthedocs.io)
[![GitHub R package version](https://img.shields.io/github/r-package/v/mingdeyu/dgpsi-R)](https://github.com/mingdeyu/dgpsi-R)

## For R users
The `R` interface to the package is available at [`dgpsi-R`](https://github.com/mingdeyu/dgpsi-R).

## A Python package for deep and linked Gaussian process emulations using stochastic imputation (SI)
`dgpsi` currently implements:

* Gaussian process emulations with separable or non-separable squared exponential and Mat&eacute;rn-2.5 kernels.
* Deep Gaussian process emulation with flexible architecture construction: 
    - multiple layers;
    - multiple GP nodes;
    - separable or non-separable squared exponential and Mat&eacute;rn2.5 kernels;
    - global input connections;
    - non-Gaussian likelihoods (Poisson, Negative-Binomial, heteroskedastic Gaussian, and categorical);
* Linked emulation of feed-forward systems of computer models:
    - linking GP emulators of deterministic individual computer models;
    - linking GP and DGP emulators of deterministic individual computer models;
* Multi-core predictions from GP, DGP, and Linked (D)GP emulators;
* Fast Leave-One-Out (LOO) cross validations for GP and DGP emulators.
* Calculations of ALM, MICE, and VIGF sequential design criterions.
* ![Feature Badge](https://img.shields.io/badge/Feature-New-orange) [Large-scale GP, DGP, and Linked (D)GP emulations.](https://github.com/mingdeyu/DGP/blob/master/demo/vecchia_SI.ipynb)
* ![Feature Badge](https://img.shields.io/badge/Feature-New-orange) [Scalable DGP classification using Stochastic Imputation.](https://github.com/mingdeyu/DGP/blob/master/demo/DGP_classification.ipynb)

## Installation
`dgpsi` currently requires Python version 3.7, 3.8, 3.9, or 3.10. The package can be installed via `pip`:

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

## Demo and documentation
Please see [demo](https://github.com/mingdeyu/DGP/tree/master/demo) for some illustrative examples of the method. The API reference 
of the package can be accessed from [https://dgpsi.readthedocs.io](https://dgpsi.readthedocs.io).

## Tips
* Since SI is a stochastic inference, in case of unsatisfactory results, you may want to try to restart the training multiple times even with initial values of hyperparameters unchanged;
* The recommended DGP structure is a two-layered one with the number of GP nodes in the first layer equal to the number of input dimensions (i.e., number of input columns) and the number of GP nodes in the second layer equal to the number of output dimensions (i.e., number of output columns) or the number of parameters in the specified likelihood.

## Contact
Please feel free to email me with any questions and feedbacks: 

Deyu Ming <[deyu.ming.16@ucl.ac.uk](mailto:deyu.ming.16@ucl.ac.uk)>.

## Research Notice
This package is part of an ongoing research initiative. For detailed information about the research aspects and guidelines for use, please refer to our [Research Notice](./RESEARCH-NOTICE.md).

## References
> [Ming, D. and Williamson, D. (2023) Linked deep Gaussian process emulation for model networks. arXiv:2306.01212.](https://arxiv.org/abs/2306.01212)

> [Ming, D., Williamson, D., and Guillas, S. (2023) Deep Gaussian process emulation using stochastic imputation. <i>Technometrics</i>. 65(2), 150-161.](https://doi.org/10.1080/00401706.2022.2124311)

> [Ming, D. and Guillas, S. (2021) Linked Gaussian process emulation for systems of computer models using Mat&eacute;rn kernels and adaptive design, <i>SIAM/ASA Journal on Uncertainty Quantification</i>. 9(4), 1615-1642.](https://epubs.siam.org/doi/abs/10.1137/20M1323771)
