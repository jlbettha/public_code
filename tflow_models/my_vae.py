import sys, os, math, time, pathlib
import numpy as np

# from matplotlib import interactive
import matplotlib.pyplot as plt

# from readDicom import *
import fnmatch
import json

# from imageio import imwrite
import random
import cv2
from scipy import ndimage as ndi
from collections import Counter
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.models import load_model
from matplotlib.path import Path
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals
import joblib
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


####
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def total_loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 15000  # 512/.1
        reconstruction_loss = K.mean(K.square(y_true - y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * K.sum(
            1.0
            + encoder_log_variance
            - K.square(encoder_mu)
            - K.exp(encoder_log_variance),
            axis=1,
        )
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(encoder_mu, encoder_log_variance)

        loss = reconstruction_loss + kl_loss
        return loss

    return total_loss_func


def my_vae_encoder(
    input_tensor,
    dropout=0.1,
    batchnorm=True,
    layer_multipliers=[1, 2, 2],
    n_filters=16,
    avg_pool=False,
    image_dim=128,
    latent_dim=32,
):
    conv1 = Conv2D(
        n_filters * layer_multipliers[0], (3, 3), activation="relu", padding="same"
    )(input_tensor)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    if batchnorm:
        conv1 = BatchNormalization()(conv1)
    if avg_pool:
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  #
    else:
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(
        n_filters * layer_multipliers[1], (3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    if batchnorm:
        conv2 = BatchNormalization()(conv2)
    if avg_pool:
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  #
    else:
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(
        n_filters * layer_multipliers[2], (3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    if batchnorm:
        conv3 = BatchNormalization()(conv3)
    if avg_pool:
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  #
    else:
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    flatten = Flatten()(pool3)
    z_mean = Dense(latent_dim, activation="relu")(flatten)

    z_log_var = Dense(latent_dim, activation="relu")(flatten)

    encoder_z_log_var_model = Model(
        inputs=[input_tensor],
        outputs=[z_mean, z_log_var],
        name="encoder_z_log_var_model",
    )

    z = Sampling(name="encoder_output")([z_mean, z_log_var])

    encoder = Model(
        inputs=[input_tensor], outputs=[z, z_mean, z_log_var], name="encoder_model"
    )

    return encoder, z_mean, z_log_var, pool3.shape


def my_vae_decoder(
    input_tensor,
    latent_dim=32,
    batchnorm=True,
    layer_multipliers=[1, 2, 2],
    n_filters=16,
    image_dim=128,
    reshape_param=(None, 16, 16, 32),
):
    z_resp = Dense(
        image_dim**2 * layer_multipliers[2] * n_filters // 64, activation="relu"
    )(input_tensor)
    z_resp = LeakyReLU(alpha=0.1)(z_resp)
    z_resp = Reshape((reshape_param[1:]))(z_resp)

    # ### Decoder
    up0 = UpSampling2D((2, 2), interpolation="bicubic")(z_resp)
    conv4 = Conv2DTranspose(
        n_filters * layer_multipliers[2], (3, 3), activation="relu", padding="same"
    )(up0)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    if batchnorm:
        conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2, 2), interpolation="bicubic")(conv4)

    conv5 = Conv2DTranspose(
        n_filters * layer_multipliers[1], (3, 3), activation="relu", padding="same"
    )(up1)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    if batchnorm:
        conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2, 2), interpolation="bicubic")(conv5)

    decoder_out = Conv2DTranspose(1, (3, 3), activation="relu", padding="same")(up2)

    return Model(inputs=[input_tensor], outputs=[decoder_out], name="decoder")


#### Set folders and paths
if __name__ == "__main__":
    # get data

    # train/val/test split

    # image pre-processing, if desired

    # initialize variables
    image_dim = 128  # Done: 128,
    feature_dim = 32
    layer_mult = [2, 3, 4]
    num_epochs, batch_size = 100, 64
    partition_size = batch_size * 8
    mode = ["train", "test"][0]
    n_z = feature_dim
    latent_dim = feature_dim
    loss = ["binary_crossentropy", "categorical_crossentropy", "mse", "mae"][2]
    opt = ["adam", "rmsprop", "sgd"][0]
    model_save_str = "./saved_models/this_vae_model.h5"

    monitor = "val_loss"
    callbacks = [
        EarlyStopping(monitor=monitor, patience=15, verbose=1),
        ModelCheckpoint(
            monitor=monitor,
            filepath=model_save_str,
            mode="auto",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    vae_input = Input(shape=(image_dim, image_dim, 1), name="VAE_input")
    encoder, z_mean, z_log_var, reshape_param = my_vae_encoder(
        vae_input,
        latent_dim=feature_dim,
        batchnorm=False,
        layer_multipliers=layer_mult,
        n_filters=16,
        image_dim=128,
    )
    encoder.summary()

    decoder_input = Input(shape=(latent_dim), name="decoder_input")
    decoder = my_vae_decoder(
        decoder_input,
        latent_dim=feature_dim,
        reshape_param=reshape_param,
        batchnorm=False,
        layer_multipliers=layer_mult,
        n_filters=16,
        image_dim=128,
    )
    decoder.summary()

    vae_encoder_output, _, _ = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)

    model = Model(vae_input, vae_decoder_output, name="VAE")
    optimizer = tf.keras.optimizers.Adam()  # learning_rate=1e-3)
    model.compile(optimizer=opt, loss=total_loss_func(z_mean, z_log_var))

    try:
        model.load_weights(model_save_str)
        print("LOADED PRETRAINED MODEL")
    except:
        print("BUILDING NEW MODEL")

    #### model batch training
    print("Begin model training...")
    for ep in range(num_epochs):
        # TODO: train vae

        pass
