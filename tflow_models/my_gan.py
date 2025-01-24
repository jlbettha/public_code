import tensorflow as tf
from keras import layers

# TODO: make convolutional like unet or similar


# Generator
def my_generator(
    input_tensor,
    num_classes,
    dropout=0.2,
    batchnorm=True,
    n_filters=16,
    avg_pool=False,
):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(28 * 28 * 1, activation="tanh"))
    model.add(layers.Reshape((28, 28, 1)))
    return model


# Discriminator
def my_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


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
    z = layers.Input(shape=(100,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model (stacked generator and discriminator)
    gan = tf.keras.Model(z, validity)
    gan.compile(
        loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
    )

    return generator, discriminator, gan
