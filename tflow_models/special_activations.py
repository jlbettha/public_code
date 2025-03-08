import tensorflow as tf


def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf


def swish(x, beta: float = 1.0):
    return x / (1 + tf.exp(-beta * x))
