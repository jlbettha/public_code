""" _unet_
Created on Fri Jul 17 17:32:34 2020
@author: jlbettha
"""

from keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    Conv2DTranspose,
    Concatenate,
)
from keras.models import Model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(
        filters=n_filters,
        activation="linear",
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    x = LeakyReLU(alpha=0.1)(x)
    if batchnorm:
        x = BatchNormalization()(x)

    # second layer
    x = Conv2D(
        filters=n_filters,
        activation="linear",
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    x = LeakyReLU(alpha=0.1)(x)
    if batchnorm:
        x = BatchNormalization()(x)

    return x


def my_unet(input_tensor, n_filters=16, dropout=0.1, batchnorm=True):
    """UNET"""

    ### Contracting Path
    c1 = conv2d_block(
        input_tensor, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(
        n_filters * 8, (3, 3), activation="linear", strides=(2, 2), padding="same"
    )(c5)
    u6 = LeakyReLU(alpha=0.1)(u6)
    if batchnorm:
        u6 = BatchNormalization()(u6)
    u6 = Concatenate()([u6, c4])
    # u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(
        n_filters * 4, (3, 3), activation="linear", strides=(2, 2), padding="same"
    )(c6)
    u7 = LeakyReLU(alpha=0.1)(u7)
    if batchnorm:
        u7 = BatchNormalization()(u7)
    u7 = Concatenate()([u7, c3])
    # u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(
        n_filters * 2, (3, 3), activation="linear", strides=(2, 2), padding="same"
    )(c7)
    u8 = LeakyReLU(alpha=0.1)(u8)
    if batchnorm:
        u8 = BatchNormalization()(u8)
    u8 = Concatenate()([u8, c2])
    # u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(
        n_filters * 1, (3, 3), activation="linear", strides=(2, 2), padding="same"
    )(c8)
    u9 = LeakyReLU(alpha=0.1)(u9)
    if batchnorm:
        u9 = BatchNormalization()(u9)
    u9 = Concatenate()([u9, c1])
    # u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation="linear")(c9)
	
    return Model(inputs=[input_tensor], outputs=[outputs])
