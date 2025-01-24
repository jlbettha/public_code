from keras.layers import (
    LSTM,
    Dense,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    Concatenate,
    TimeDistributed,
)
from keras.models import Model


def LSTM_classification(
    input_tensor, n_classes, n_nodes=64, bidirectional=False, return_sequences=False
):
    """LSTM classification model

    Args:
        input_tensor (_type_): Input(input.shape)
        n_classes (int): number of classes to predict.
        n_nodes (int, optional): LSTM nodes per layer. Defaults to 64.
        bidirectional (bool, optional): Bidirectional/acausal LSTM if True. Defaults to False.
        return_sequences (bool, optional): Seq-to-seq if True. Defaults to False.

    Returns:
        Model: classification decision model (one-hot encoded)
    """
    LSTM_model = LSTM(
        n_nodes, input_shape=input_tensor.shape, return_sequences=return_sequences
    )(input_tensor)

    # Birdirectional LSTM
    if bidirectional:
        model_backwards = LSTM(
            n_nodes, return_sequences=return_sequences, go_backwards=True
        )(input_tensor)
        # model = Merge(mode="concat")([model, model_backwards]) # old version of keras
        LSTM_model = Concatenate()([LSTM_model, model_backwards])

    dense1 = Dense(2 * n_nodes, activation="linear")(LSTM_model)
    dense1 = LeakyReLU(alpha=0.1)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(n_nodes, activation="linear")(dense1)
    dense2 = LeakyReLU(alpha=0.1)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.10)(dense2)

    model_out = TimeDistributed(Dense(n_classes, activation="softmax"))(dense2)
	
    return Model(inputs=input_tensor, outputs=model_out)


def LSTM_regression(
    input_tensor, n_nodes=64, bidirectional=False, return_sequences=False
):
    """LSTM regression model

    Args:
        input_tensor (_type_): Input(input.shape)
        n_nodes (int, optional): LSTM nodes per layer. Defaults to 64.
        bidirectional (bool, optional): Bidirectional/acausal LSTM if True. Defaults to False.
        return_sequences (bool, optional): Seq-to-seq if true. Defaults to False.

    Returns:
        Model: regression estimation model in [0,1]
    """
    LSTM_model = LSTM(
        n_nodes, input_shape=input_tensor.shape, return_sequences=return_sequences
    )(input_tensor)

    # Birdirectional LSTM
    if bidirectional:
        model_backwards = LSTM(
            n_nodes, return_sequences=return_sequences, go_backwards=True
        )(input_tensor)
        # model = Merge(mode="concat")([model, model_backwards]) # old version of keras
        LSTM_model = Concatenate()([LSTM_model, model_backwards])

    dense1 = Dense(2 * n_nodes, activation="linear")(LSTM_model)
    dense1 = LeakyReLU(alpha=0.1)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)

    dense2 = Dense(n_nodes, activation="linear")(dense1)
    dense2 = LeakyReLU(alpha=0.1)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.10)(dense2)

    model_out = TimeDistributed(Dense(1, activation="linear"))(dense2)
    return Model(inputs=input_tensor, outputs=model_out)
