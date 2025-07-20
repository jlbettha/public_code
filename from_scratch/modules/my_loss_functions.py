"""
Created on Thu Feb 18 20:08:25 2021

@author: jlb235
"""

import keras.backend as K
import numpy as np
import tensorflow as tf


def mean_squared_error(ytrue, ypred):
    return (ypred - ytrue) ** 2 / np.size(ytrue)


def mse_derivative(ytrue, ypred):
    return 2 * (ypred - ytrue) / np.size(ytrue)


def vae_loss(mean, var) -> float:
    """
    _summary_

    Args:
        mean (_type_): _description_
        var (_type_): _description_

    Returns:
        float: _description_

    """

    def kl_loss(mean, log_var):
        return -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=1)

    def mse_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

    def loss(y_true, y_pred):
        log_var = K.log(var)
        r_loss = mse_loss(y_true, y_pred)
        k_loss = kl_loss(mean, log_var)
        return 1000 * r_loss + k_loss

    return loss


def focal_loss(gamma: float = 2.0, alpha: float = 4.0) -> float:
    """
    Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        gamma (float): default: 2.0
        alpha (float): default: 4.0

    Returns:
        float: loss.

    """

    def loss(y_true, y_pred):
        epsilon = 1.0e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        # y_true = K.flatten(y_true)
        # y_pred = K.flatten(y_pred)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1.0, model_out), float(gamma)))
        fl = tf.multiply(float(alpha), tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return loss


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25) -> float:
    """
    Binary form of focal loss.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf

    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))

    return loss


def categorical_focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> float:
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
        FL = ∑_c=1,m  [-alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)]
        where m = number of classes, c = class and o = observation
    Parameters:
        alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
        categories/labels, the size of the array needs to be consistent with the number of classes.
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    """

    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = float(alpha) * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    return loss


def weighted_categorical_crossentropy(weights: np.ndarray[float]) -> float:
    """
    _summary_

    Args:
        weights (np.ndarray[float]): _description_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        loss = y_true * K.log(y_pred) * K.variable(weights)
        return -K.sum(loss, -1)

    return loss


def weighted_categorical_crossentropy2(weights: np.ndarray[float]) -> float:
    """
    _summary_

    Args:
        weights (np.ndarray[float]): _description_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        k_weights = K.constant(weights)
        if not tf.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * k_weights, axis=-1)

    return loss


def weighted_pixelwise_crossentropy(class_weights: np.ndarray[float]) -> float:
    """
    _summary_

    Args:
        class_weights (np.ndarray[float]): _description_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        return -K.sum(y_true * K.log(y_pred) * class_weights)

    return loss


def dice_loss() -> float:
    """
    _summary_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2.0 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true + y_pred)
        return 1 - numerator / denominator

    return loss


def dice_loss2(smooth: float = 1.0) -> float:
    """
    _summary_

    Args:
        smooth (float, optional): _description_. Defaults to 1.

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return loss


def tversky_loss(smooth: float = 1, alpha: float = 0.7) -> float:
    """
    _summary_

    Args:
        smooth (float, optional): _description_. Defaults to 1.
        alpha (float, optional): _description_. Defaults to 0.7.

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    return loss


def focal_tversky_loss(gamma: float = 2.0, smooth: float = 1.0, alpha: float = 0.7) -> float:
    """
    _summary_

    Args:
        gamma (float, optional): _description_. Defaults to 2.0.
        smooth (float, optional): _description_. Defaults to 1.0.
        alpha (float, optional): _description_. Defaults to 0.7.

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        tv = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        return K.pow((1 - tv), gamma)

    return loss


def ssim_loss() -> float:
    """
    _summary_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        """
        Structural Similarity Index (SSIM) loss
        """
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)

    return loss


def jacard_loss() -> float:
    """
    Intersection-Over-Union (IoU), also known as the Jaccard loss

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        jacard_similarity = intersection / union
        return 1 - jacard_similarity

    return loss


def log_loss() -> float:
    """
    _summary_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-12, 1 - 1e-12)
        error = y_true * tf.math.log(y_pred + 1e-12)(1 - y_true) * tf.math.log(1 - y_pred + 1e-12)
        return -error

    return loss


def weighted_bce() -> float:
    """
    _summary_

    Returns:
        float: _description_

    """

    def loss(y_true, y_pred):
        weights = (y_true * 50.0) + 1.0
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.mean(bce * weights)

    return loss


def unet3p_hybrid_loss() -> float:
    """
    Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy pixel, patch, and map-level,
    which is able to capture both large-scale and fine structures with clear boundaries.
    """

    def weighted_bce(y_true, y_pred):
        weights = (y_true * 50.0) + 1.0
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.mean(bce * weights)

    def jacard_val(y_true, y_pred):
        """
        Intersection-Over-Union (IoU), also known as the Jaccard index
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        jacard_similarity = intersection / union
        return 1 - jacard_similarity

    def ftv_val(y_true, y_pred, gamma=2.0, smooth=1.0, alpha=0.7):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        tv = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        return K.pow((1 - tv), gamma)

    def ssim_val(y_true, y_pred):
        val = int(np.sqrt(y_true.shape[1]))
        return 1 - tf.image.ssim(
            tf.reshape(y_true, [-1, val, val, 1]),
            tf.reshape(y_pred, [-1, val, val, 1]),
            max_val=1,
        )

    def loss(y_true, y_pred):
        wce = weighted_bce(y_true, y_pred)
        ftv = ftv_val(y_true, y_pred)
        ssim = ssim_val(y_true, y_pred)
        jac = jacard_val(y_true, y_pred)
        return 0.6 * ftv + 0.1 * ssim + 0.1 * jac + 0.2 * wce

    return loss


def main() -> None:
    print("my_loss_functions.py is a module")


if __name__ == "__main__":
    main()
