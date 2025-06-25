"""_summary_"""

from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    Conv2DTranspose,
    UpSampling2D,
    Flatten,
    Reshape,
    Dense,
)
from keras.models import Model
from keras import regularizers


def autoencoder(
    input_tensor, dropout=0.2, batchnorm=True, n_filters=16, avg_pool=False
):
    """_summary_

    Args:
        input_tensor (_type_): _description_
        dropout (float, optional): _description_. Defaults to 0.2.
        batchnorm (bool, optional): _description_. Defaults to True.
        n_filters (int, optional): _description_. Defaults to 16.
        avg_pool (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    ### Encoder
    conv1 = Conv2D(n_filters * 1, (3, 3), activation="linear", padding="same")(
        input_tensor
    )
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    if batchnorm:
        conv1 = BatchNormalization()(conv1)
    if avg_pool:
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  #
    else:
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation="linear", padding="same")(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    if batchnorm:
        conv2 = BatchNormalization()(conv2)
    if avg_pool:
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  #
    else:
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(
        n_filters * 4,
        (3, 3),
        activation="linear",
        activity_regularizer=regularizers.l1(0.001),
        padding="same",
    )(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    if batchnorm:
        conv3 = BatchNormalization()(conv3)
    if avg_pool:
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  #
    else:
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    ### Decoder
    up0 = UpSampling2D((2, 2), interpolation="bicubic")(pool3)
    conv4 = Conv2DTranspose(n_filters * 4, (3, 3), activation="linear", padding="same")(
        up0
    )
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    if batchnorm:
        conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2, 2), interpolation="bicubic")(conv4)

    conv5 = Conv2DTranspose(n_filters * 2, (3, 3), activation="linear", padding="same")(
        up1
    )
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    if batchnorm:
        conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2, 2), interpolation="bicubic")(conv5)

    decoded = Conv2DTranspose(1, (3, 3), activation="linear", padding="same")(up2)

    return Model(inputs=[input_tensor], outputs=[decoded])


def autoencoder_feature_bottleneck(
    input_tensor,
    dropout=0.1,
    batchnorm=True,
    n_filters=16,
    avg_pool=False,
    image_dim=256,
):
    """_summary_

    Args:
        input_tensor (_type_): _description_
        dropout (float, optional): _description_. Defaults to 0.1.
        batchnorm (bool, optional): _description_. Defaults to True.
        n_filters (int, optional): _description_. Defaults to 16.
        avg_pool (bool, optional): _description_. Defaults to False.
        image_dim (int, optional): _description_. Defaults to 256.

    Returns:
        _type_: _description_
    """

    ### Encoder
    conv1 = Conv2D(n_filters * 4, (3, 3), activation="linear", padding="same")(
        input_tensor
    )
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    if batchnorm:
        conv1 = BatchNormalization()(conv1)
    if avg_pool:
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    else:
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation="linear", padding="same")(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    if batchnorm:
        conv2 = BatchNormalization()(conv2)
    if avg_pool:
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  #
    else:
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(n_filters * 1, (3, 3), activation="linear", padding="same")(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    if batchnorm:
        conv3 = BatchNormalization()(conv3)
    if avg_pool:
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  #
    else:
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    ### Bottleneck, encoder output
    bottle1 = Flatten()(pool3)
    bottle2 = Dense(
        128, activation="linear", activity_regularizer=regularizers.l1(0.001)
    )(bottle1)
    bottle2 = LeakyReLU(alpha=0.1)(bottle2)

    ### Decoder input
    bottle3 = Dense(image_dim**2 * n_filters // 64, activation="linear")(bottle2)
    bottle3 = LeakyReLU(alpha=0.1)(bottle3)
    bottle3 = Reshape((image_dim // 8, image_dim // 8, n_filters))(bottle3)

    ### Decoder
    up0 = UpSampling2D((2, 2), interpolation="bicubic")(bottle3)
    conv4 = Conv2DTranspose(n_filters * 1, (3, 3), activation="linear", padding="same")(
        up0
    )  # 7 x 7 x 128
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    if batchnorm:
        conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2, 2), interpolation="bicubic")(conv4)

    conv5 = Conv2DTranspose(n_filters * 2, (3, 3), activation="linear", padding="same")(
        up1
    )
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    if batchnorm:
        conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2, 2), interpolation="bicubic")(conv5)

    # decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(up2)
    decoded = Conv2DTranspose(1, (3, 3), activation="linear", padding="same")(up2)
    return Model(inputs=[input_tensor], outputs=[decoded])
