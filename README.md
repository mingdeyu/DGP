# Deep and Linked Gaussian Process Emulation using Stochastic Imputation
The package `dgpsi` implements inference of both deep and linked Gaussian process emulation using stochastic imputation. 

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

Please see [demo](demo/) for some illustrative examples of the method. Detailed descriptions on how to use the package can be found in scripts contained in [dgpsi](dgpsi/).

## Installation
After cloning the repo, type the following in the same directory of `setup.py`:

```bash
pip install .
```

to install the code and its required dependencies.

## Tips
* Since SI is a stochastic inference, in case of unsatisfactory results, you may want to try to restart the training multiple times even with initial values of hyperparameters unchanged;
* The recommended DGP structure is a two-layered one with the number of GP nodes in the first layer equal to the number of input dimensions (i.e., number of input columns) and the number of GP nodes in the second layer equal to the number of output dimensions (i.e., number of output columns) or the number of parameters in the specified likelihood.

## Built with
The package is built under `Python 3.7.3` with following packages:
* `numpy 1.18.2`;
* `numba 0.51.2`;
* `matplotlib 3.2.1`;
* `tqdm 4.50.2`;
* `scikit-learn 0.22.0`;
* `scipy 1.4.1`;
* `dill 0.3.2`;
* `pathos 0.2.9`;
* `psutil 5.8.0`.

## Contact
Please feel free to email me with any questions and feedbacks: 

Deyu Ming <[deyu.ming.16@ucl.ac.uk](mailto:deyu.ming.16@ucl.ac.uk)>.

## References
> [Ming, D., Williamson, D., and Guillas, S. (2021) Deep Gaussian process emulation using stochastic imputation.](https://arxiv.org/abs/2107.01590)

> [Ming, D. and Guillas, S. (2021) Linked Gaussian process emulation for systems of computer models using Mat&eacute;rn kernels and adaptive design, <i>SIAM/ASA Journal on Uncertainty Quantification</i>. 9(4), 1615-1642.](https://epubs.siam.org/doi/abs/10.1137/20M1323771)
