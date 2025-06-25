# public_code/ 

Ongoing refactoring project being populated with tools I've built over the years. Because
I am unable to share many of my work products publicly, I'm taking time to implement certain
shareable aspects using different contexts & data.

## Contents

### from_scratch/  

Manual implementations for fundamental ML methods and concepts.

### data_compression/

* DCT compression
* k-means compression
* k-medians compression
* orthonormal compression

### image_processing/

* angio image/video tests
* pdf 2 text
* image 2 text


### image_segmentation/

* content in-progress


### image_classification/

* content in-progress


### optical_flow/

* horn-schunck opflow
* lucas-kanade opflow
* in-built keypoint tracker
* angiogram videos showing tracking of coronary arterial blockage


### my_modules/

* image feature extraction tools
* distance functions: point to point, point to distribution, and distr to distr
* helper/util functions
* a basic model builder kit (from scratch)
* loss functions
* information theory tools
* image quality assessment tools
* all modules are also in ./from_scratch/


### temporal_sequential/

* mouse tracker: kalman and unscented kalman
* audio, time-series prediction/forecast, ML + DSP
* temporal conv net, lstm, etc (models may also be in /tflow_models/ or /torch_models/)
* optical flow
* activity recognition


### tflow_models/  

Tensorflow models and implementations

Done:
* cnn, lstm, autoencoder, unet, vae, gan, attn and multihead, transformer, vision transformer

To-do:
* vq vae
* vq gan
* beta vae (next up)


### torch_models/

* content in-progress
* PyTorch versions of certain models found in /tflow_models

