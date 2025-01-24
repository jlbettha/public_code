import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    Dropout,
    Embedding,
    MultiHeadAttention,
)
from keras.models import Model


def my_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def my_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    num_classes,
    dropout=0,
    mlp_units=None,
    mlp_dropout=0,
):
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = my_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    if mlp_units:
        for units in mlp_units:
            x = Dense(units, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


if __name__ == "__main__":
    input_shape = (100, 64)  # Example input shape
    head_size = 256
    num_heads = 4
    ff_dim = 256
    num_transformer_blocks = 4
    num_classes = 10
    dropout = 0.1
    mlp_units = [128]
    mlp_dropout = 0.1

    model = my_transformer_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        num_classes,
        dropout,
        mlp_units,
        mlp_dropout,
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
