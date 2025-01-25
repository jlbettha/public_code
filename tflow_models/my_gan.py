"""_summary_"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    LeakyReLU,
    BatchNormalization,
    Dense,
    Input,
    Reshape,
)

# TODO: make convolutional like unet or similar


# Generator
def my_generator(input_tensor, dropout=0.2, batchnorm=True):
    d1 = Dense(256, activation="linear")(input_tensor)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1 = BatchNormalization(momentum=0.8)(d1)
    d2 = Dense(512, activation="linear")(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d3 = Dense(1024, activation="linear")(d2)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = BatchNormalization(momentum=0.8)(d3)
    out = Dense(28 * 28 * 1, activation="tanh")(d3)
    out = Reshape((28, 28, 1))(out)

    return Model(inputs=[input_tensor], outputs=[out], name="generator_model")


# Discriminator
def my_discriminator(input_tensor, dropout=0.2, batchnorm=True):
    d1 = Dense(512, activation="linear")(input_tensor)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d2 = Dense(256, activation="linear")(d1)
    d2 = LeakyReLU(alpha=0.2)(d2)
    out = Dense(1, activation="sigmoid")(d2)

    return Model(inputs=[input_tensor], outputs=[out], name="discriminator_model")


# Compile the discriminator
def my_gan():
    discriminator = my_discriminator()
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=["accuracy"],
    )

    # Build the generator
    generator = my_generator()

    # The generator takes noise as input and generates images
    z = Input(shape=(100,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model (stacked generator and discriminator)
    gan = Model(z, validity)
    gan.compile(
        loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
    )

    return generator, discriminator, gan
