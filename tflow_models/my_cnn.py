"""_summary_"""

from keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPooling2D,
)
from keras.models import Model


def my_cnn(
    input_tensor,
    num_classes,
    dropout=0.2,
    *,
    batchnorm=True,
    n_filters=16,
    avg_pool=False,
):
    """
    _summary_

    Args:
        input_tensor (_type_): _description_
        num_classes (_type_): _description_
        dropout (float, optional): _description_. Defaults to 0.1.
        batchnorm (bool, optional): _description_. Defaults to True.
        n_filters (int, optional): _description_. Defaults to 16.
        loss (str, optional): _description_. Defaults to "categorical_crossentropy".
        optimizer (str, optional): _description_. Defaults to "adam".
        avg_pool (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    """
    #### Encoder
    conv1 = Conv2D(n_filters * 2, (3, 3), activation="linear", padding="same")(input_tensor)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    if batchnorm:
        conv1 = BatchNormalization()(conv1)

    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1) if avg_pool else MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(n_filters * 4, (3, 3), activation="linear", padding="same")(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    if batchnorm:
        conv2 = BatchNormalization()(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) if avg_pool else MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(n_filters * 8, (3, 3), activation="linear", padding="same")(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    if batchnorm:
        conv3 = BatchNormalization()(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3) if avg_pool else MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)
    encoder_out = Flatten()(pool3)

    #### MLP layers
    dense1 = Dense(128, activation="linear")(encoder_out)
    dense1 = LeakyReLU(alpha=0.1)(dense1)
    # dense1 = Dropout(dropout)(dense1)

    # dense1 = Dense(32, activation = 'relu')(dense1)
    one_hot1 = Dense(num_classes, activation="softmax")(dense1)

    return Model(inputs=[input_tensor], outputs=[one_hot1])
