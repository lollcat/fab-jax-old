# Flow Annealed Importance Sampling Boostrap (FAB) implementation in JAX
 - See corresponding paper [here](https://arxiv.org/abs/2208.01893).
 - See the pytorch implementation [here](https://github.com/lollcat/fab-torch). 
 - The SNR analysis of the FAB gradient estimator is [here](https://github.com/lollcat/fab-jax-old/tree/main/examples_fabjax/visualisation_gradient_estimators).
 - I am currently working on a new version improved version of the jax implementation of FAB. 


## Examples
**Double Well Boltzmann distribution samples vs contours** (see [notebook](examples_fabjax/double_well.ipynb))
![Double Well samples vs contours](examples_fabjax/images/double_well.png)

**Gaussian Mixture Model samples vs contours** (see [notebook](examples_fabjax/gmm.ipynb))
![Gaussian Mixture Model samples vs contours](examples_fabjax/images/gmm.png)
