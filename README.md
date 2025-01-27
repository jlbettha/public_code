# public_code/ 

Ongoing refactoring project being populated with tools I've built over the years. Because
I am unable to share many of my work products publicly, I'm taking time to implement certain
shareable aspects using different contexts & data.

## Contents

### from_scratch/  

Manual implementations for fundamental ML methods and concepts.

### image_processing/

* angio image/video tests
* pdf 2 text
* image 2 text


### image_segmentation/

* content in-progress


### optical_flow/

* horn-schunck opflow
* lucas-kanade opflow
* in-built keypoint tracker
* angiogram videos showing tracking of coronary arterial blockage


### my_modules/

* image feature extraction
* distance functions: point to point, point to distribution, and distr to distr
* helper/util functions
* a basic model builder kit (from scratch)


### nlp_language_models/

* working on word embedding, small language model, and translation


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

* PyTorch versions of many models found in /tflow_models


### raspberrypi5_apps/

* microphone array beamforming, sound source localization 
* sound source separation
* speaker identification and speech-to-text
* stereoscopic vision for 3d scene reconstruct
* 3d from 2d imaging model, trained from stereoscopic videos
* IR heat mapping of images (stream/record FLIR-ONE from phone)
* Database for queries from laptop, desktop queries rasp. pi for gpu model training and RAG

### model_interpretability/

* content in-progress

### reinforcement_learning/

* coming soon

### scene_reconstruction/

* coming soon
