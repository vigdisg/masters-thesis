# Training MMD networks

MMDNet_standard_normal.py trains [MMD network](https://arxiv.org/abs/1502.02761) (with RBF kernel) to generate from a standard normal distribution (chapter 7.1 in the thesis). We run the code seperately for each N = 50,100,1000 and epochs = 10,80,150.

MNIST.ipynb (framework does not include an autoencoder), MNIST_AE.ipynb (seperately trained autoencoder included) and MNIST_AE_dropout.ipynb (autoencoder has dropout in encoding layers) train MMD networks (with RBF kernel) to generate from MNIST (chapter 7.2 in the thesis).

Simulations1.ipynb, Simulations2.ipynb and Simulations3.ipynb train MMD networks (with RBF kernel, laplacian kernel and RQ kernel) and energy distance networks to generate from three conditional distributions (chapter 7.3 in the thesis).