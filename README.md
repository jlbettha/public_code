# Repository: `public_code/`

Ongoing refactoring project for converting tools I've built over the years to python. Current repository iteration is focused on being JIT-friendly (via `numba`) where possible, formatted by __Ruff__ (using `.ruff.toml` and `.pre-commit-config.yaml`), and package-managed by Astral's amazing __uv__ tool.

## Contents

### Data Compression: [`./data_compression/`](https://github.com/jlbettha/public_code/tree/main/data_compression/)
* DCT compression
* _k_-means compression
* _k_-medians compression
* Orthonormal compression


### From Scratch: [`./from_scratch/`](https://github.com/jlbettha/public_code/tree/main/from_scratch/)
* My manual implementations for many fundamental ML methods and concepts.
* __Projection methods:__ Singular Value Decomposition (SVD), Principal Component Analysis (PCA), and Linear Discriminant Analysis (LDA)
* __Clustering methods:__ _k_-means clustering, _k_-Gaussian mixture model, and hierarchical clustering
* __Optimization:__ gradient descent and stochastic gradient descent
* __Regression methods:__ linear regression with ordinary least squares and  regularized ($\ell_0$, $\ell_1$, and $\ell_2$), logitic regression, and polynomial regression with regularization ($\ell_0$, $\ell_1$, and $\ell_2$)
* __Classification methods:__ decision trees, support vector machine (SVM) with Gaussian radial basis function, and artificial neural network from multi-layer perceptrons (MLP)
* __Partitioning functions:__ unweighted and weighted options
* __Sorting functions:__ insertion sort, selection sort, heap sort, tim sort, radix sort, bubble sort, and counting sort
* __Search methods:__ A\*, beam, breadth-first, and depth-first search functions (not yet implemented)


### Image Classification/: [`./image_classification/`](https://github.com/jlbettha/public_code/tree/main/image_classification/)
* __Future content:__ in-progress and unavailable


### Image_processing: [`./image_processing/`](https://github.com/jlbettha/public_code/tree/main/image_processing/)
* Angiogram imaging/video tests
* Bi-lateral filter


### Image Segmentation: [`./image_segmentation/`](https://github.com/jlbettha/public_code/tree/main/image_segmentation/)
* __Future content:__ in-progress and unavailable


### My personal modules and utilities: [`./my_modules/`](https://github.com/jlbettha/public_code/tree/main/my_modules/)
* Image feature extraction tools
* Distance functions: point-to-point, point-to-distribution, and distribution-to-distribution
* Helper/utility functions
* Loss functions (Keras style)
* Activation functions
* Information Theory tools
* Image Quality Assessment tools
* Kalman Filter and Unscented Kalman Filter
* All modules are also in [`./from_scratch/`](https://github.com/jlbettha/public_code/tree/main/from_scratch/)


### Optical Flow: [`./optical_flow/`](https://github.com/jlbettha/public_code/tree/main/optical_flow/)
* Horn-Schunck optical flow
* Lucas-Kanade optical flow
* OpenCV keypoint tracker
* Angiogram videos showing tracking of coronary arterial blockage


### Tensorflow model implementations: [`./tflow_models/`](https://github.com/jlbettha/public_code/tree/main/tflow_models/)
* __Future content:__ in-progress and unavailable
* __Done:__ CNN, LSTM, Autoencoder, UNet, VAE, $\beta$-VAE, GAN, Atention and Multi-head Attention, Transformer, Vision Transformer (ViT)
* __To-do:__ VQ VAE, VQ GAN


### PyTorch model implementations: [`./torch_models/`](https://github.com/jlbettha/public_code/tree/main/torch_models/)
* __Future content:__ in-progress and unavailable
* PyTorch implementations of the same models as in [`./tflow_models`](https://github.com/jlbettha/public_code/tree/main/tflow_models/)
