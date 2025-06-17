"""_summary_"""

import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Reshape,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Flatten,
)


def my_vision_transformer(
    input_shape,
    num_classes,
    patch_size=16,
    num_patches=196,
    projection_dim=64,
    num_heads=4,
    transformer_units=[128, 64],
    mlp_head_units=[2048, 1024],
):
    inputs = Input(shape=input_shape)

    # Create patches
    patches = Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    patches = Reshape((num_patches, projection_dim))(patches)

    # Add positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positional_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)(
        positions
    )
    encoded_patches = patches + positional_embedding

    # Transformer blocks
    for _ in range(2):
        # Layer normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)
        # Skip connection 1
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = Dense(transformer_units[1], activation=tf.nn.gelu)(x3)
        # Skip connection 2
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dense(mlp_head_units[0], activation=tf.nn.gelu)(representation)
    representation = Dense(mlp_head_units[1], activation=tf.nn.gelu)(representation)

    # Output layer
    logits = Dense(num_classes)(representation)

    # Create the Keras model
    model = Model(inputs=inputs, outputs=logits)
    return model


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 10
    model = my_vision_transformer(
        input_shape,
        num_classes,
        patch_size=16,
        num_patches=196,
        projection_dim=64,
        num_heads=4,
        transformer_units=[128, 64],
        mlp_head_units=[2048, 1024],
    )
    model.summary()
