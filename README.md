# Flow Annealed Importance Sampling Boostrap (FAB) implementation in JAX
See corresponding paper [here](https://arxiv.org/abs/2111.11510).

This repo is still a work in progress, the ```fab``` folder and ```examples``` have some working 
illustrative problems.

## Install
```
pip install git+https://github.com/lollcat/fab-jax.git
```

## Examples
**Double Well Boltzmann distribution samples vs contours** (see [notebook](examples_fabjax/double_well.ipynb))
![Double Well samples vs contours](examples_fabjax/images/double_well.png)

**Gaussian Mixture Model samples vs contours** (see [notebook](examples_fabjax/gmm.ipynb))
![Gaussian Mixture Model samples vs contours](examples_fabjax/images/gmm.png)