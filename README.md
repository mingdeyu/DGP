# Deep Gaussian Process Emulation using Stochastic Imputation
The package `dgpsi` implements inference of deep Gaussian process emulation using stochastic imputation. 

## Key features
`dgpsi` currently has the following features:

* Flexible deep Gaussian process architecture construction: 
    - multiple layers;
    - multiple GP nodes;
    - separable or non-separable squared exponential and Mat&eacute;rn2.5 kernels;
    - global input connections;
* Multiple emulations with missingness:
    - emulation with multiple output dimensions and some dimensions have partially missing output data points;
* Partially observed I/O of internal GP nodes:
    - internal GP nodes can have partially observed outputs to incorporate some prior knowledge about the DGP structure;
* Emulation of feed-forward systems of computer models:
    - linking GP emulators of individual computer models;
    - linking GP and DGP emulators of individual computer models.

Please see [synthetic_examples](synthetic_examples/) for some illustrative examples of the method. Detailed descriptions on how to use the package can be found in scripts contained in [dgpsi](dgpsi/).

## Installation
After cloning the repo, type the following in the same directory of `setup.py`:

```bash
pip install .
```

to install the code and its required dependencies.

## Built with
The package is built under `Python 3.7.3` with following packages:
* `numpy 1.18.2`;
* `numba 0.51.2`;
* `matplotlib 3.2.1`;
* `tqdm 4.50.2`;
* `scikit-learn 0.22.0`;
* `scipy 1.4.1`.

## Contact
Please feel free to email me with any questions and feedbacks: 

Deyu Ming <[deyu.ming.16@ucl.ac.uk](mailto:deyu.ming.16@ucl.ac.uk)>.

## References
The code implements the methods in the following paper for linked GP emulation:

> [Ming, D. and Guillas, S. Linked Gaussian process emulation for systems of computer models using Mat&eacute;rn kernels and adaptive design, 2021.](https://arxiv.org/abs/1912.09468)
