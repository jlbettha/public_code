"""
Created on Thu Feb 18 20:08:25 2021

@author: jlb235
"""

import numpy as np

# import tensorflow as tf
# import tensorflow.keras.backend as K


def mean_squared_error(ytrue, ypred):
    return (ypred - ytrue) ** 2 / np.size(ytrue)


def mse_derivative(ytrue, ypred):
    return 2 * (ypred - ytrue) / np.size(ytrue)


# def mse_loss(y_true, y_pred):
#     loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
#     return 1000 * loss


# def kl_loss(mean, log_var):
#     loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=1)
#     return loss


# def vae_loss(y_true, y_pred, mean, var):
#     log_var = K.log(var)
#     r_loss = mse_loss(y_true, y_pred)
#     k_loss = kl_loss(mean, log_var)
#     return r_loss + k_loss


# def focal_loss(gamma=2.0, alpha=4.0):

#     gamma = float(gamma)
#     alpha = float(alpha)

#     def focal_loss_fixed(y_true, y_pred):
#         """Focal loss for multi-classification
#         FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
#         Notice: y_pred is probability after softmax
#         gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
#         d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
#         Focal Loss for Dense Object Detection
#         https://arxiv.org/abs/1708.02002

#         Arguments:
#             y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
#             y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

#         Keyword Arguments:
#             gamma {float} -- (default: {2.0})
#             alpha {float} -- (default: {4.0})

#         Returns:
#             [tensor] -- loss.
#         """
#         epsilon = 1.0e-9
#         y_true = tf.convert_to_tensor(y_true, tf.float32)
#         y_pred = tf.convert_to_tensor(y_pred, tf.float32)
#         # y_true = K.flatten(y_true)
#         # y_pred = K.flatten(y_pred)

#         model_out = tf.add(y_pred, epsilon)
#         ce = tf.multiply(y_true, -tf.log(model_out))
#         weight = tf.multiply(y_true, tf.pow(tf.subtract(1.0, model_out), gamma))
#         fl = tf.multiply(alpha, tf.multiply(weight, ce))
#         reduced_fl = tf.reduce_max(fl, axis=1)
#         return tf.reduce_mean(reduced_fl)

#     return focal_loss_fixed


# def binary_focal_loss(gamma=2.0, alpha=0.25):
#     """
#     Binary form of focal loss.
#       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
#       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     Usage:
#      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """

#     def binary_focal_loss_fixed(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred:  A tensor resulting from a sigmoid
#         :return: Output tensor.
#         """
#         y_true = tf.cast(y_true, tf.float32)
#         # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
#         epsilon = K.epsilon()
#         # Add the epsilon to prediction value
#         # y_pred = y_pred + epsilon
#         # Clip the prediciton value
#         y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
#         # Calculate p_t
#         p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
#         # Calculate alpha_t
#         alpha_factor = K.ones_like(y_true) * alpha
#         alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
#         # Calculate cross entropy
#         cross_entropy = -K.log(p_t)
#         weight = alpha_t * K.pow((1 - p_t), gamma)
#         # Calculate focal loss
#         loss = weight * cross_entropy
#         # Sum the losses in mini_batch
#         loss = K.mean(K.sum(loss, axis=1))
#         return loss

#     return binary_focal_loss_fixed


# def categorical_focal_loss(alpha, gamma=2.0):
#     """
#     Softmax version of focal loss.
#     When there is a skew between different categories/labels in your data set, you can try to apply this function as a
#     loss.
#            m
#       FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
#       categories/labels, the size of the array needs to be consistent with the number of classes.
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
#     Usage:
#      model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """

#     alpha = np.array(alpha, dtype=np.float32)

#     def categorical_focal_loss_fixed(y_true, y_pred):
#         # Clip the prediction value to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

#         # Calculate Cross Entropy
#         cross_entropy = -y_true * K.log(y_pred)

#         # Calculate Focal Loss
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

#         # Compute mean loss in mini_batch
#         return K.mean(K.sum(loss, axis=-1))

#     return categorical_focal_loss_fixed


# def weighted_categorical_crossentropy(weights):
#     weights = K.variable(weights)

#     def loss(y_true, y_pred):
#         # scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # clip to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss

#     return loss


# def weighted_categorical_crossentropy2(weights):
#     def loss(y_true, y_pred):
#         Kweights = K.constant(weights)
#         if not K.is_tensor(y_pred):
#             y_pred = K.constant(y_pred)
#         y_true = K.cast(y_true, y_pred.dtype)
#         return K.categorical_crossentropy(y_true, y_pred) * K.sum(
#             y_true * Kweights, axis=-1
#         )

#     return loss


# def weighted_pixelwise_crossentropy(class_weights):
#     def loss(y_true, y_pred):
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
#         return -K.sum(y_true * K.log(y_pred) * class_weights)

#     return loss


# def dice_loss():
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.math.sigmoid(y_pred)
#         numerator = 2.0 * K.sum(y_true * y_pred)
#         denominator = K.sum(y_true + y_pred)
#         return 1 - numerator / denominator

#     return loss


# def dice_loss2():
#     def loss(y_true, y_pred, smooth=1):
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#         intersection = K.sum(y_true_f * y_pred_f)
#         return 1 - (2.0 * intersection + smooth) / (
#             K.sum(y_true_f) + K.sum(y_pred_f) + smooth
#         )

#     return loss


# def tversky_loss():
#     def loss(y_true, y_pred, smooth=1, alpha=0.7):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = K.sum(y_true_pos * y_pred_pos)
#         false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#         false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
#         return 1 - (true_pos + smooth) / (
#             true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
#         )

#     return loss


# def focal_tversky_loss():
#     def loss(y_true, y_pred, gamma=2.0, smooth=1.0, alpha=0.7):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = K.sum(y_true_pos * y_pred_pos)
#         false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#         false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
#         tv = (true_pos + smooth) / (
#             true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
#         )
#         return K.pow((1 - tv), gamma)

#     return loss


# def ssim_loss():
#     def loss(y_true, y_pred):
#         """
#         Structural Similarity Index (SSIM) loss
#         """
#         return 1 - tf.image.ssim(y_true, y_pred, max_val=1)

#     return loss


# def jacard_loss():
#     def loss(y_true, y_pred):
#         """
#         Intersection-Over-Union (IoU), also known as the Jaccard loss
#         """
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)

#         intersection = K.sum(y_true_f * y_pred_f)
#         union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
#         jacard_similarity = intersection / union
#         return 1 - jacard_similarity

#     return loss


# def ftv_val(y_true, y_pred, gamma=2.0, smooth=1.0, alpha=0.7):
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#     false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
#     tv = (true_pos + smooth) / (
#         true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
#     )
#     return K.pow((1 - tv), gamma)


# def ssim_val(y_true, y_pred):
#     val = int(np.sqrt(y_true.shape[1]))
#     # y_true.reshape(-1,512,512,1)

#     return 1 - tf.image.ssim(
#         tf.reshape(y_true, [-1, val, val, 1]),
#         tf.reshape(y_pred, [-1, val, val, 1]),
#         max_val=1,
#     )


# def log_loss(y_true, y_pred):
#     y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
#     error = y_true * tf.log(y_pred + 1e-7)(1 - y_true) * tf.log(1 - y_pred + 1e-7)
#     return -error


# def weighted_bce(y_true, y_pred):
#     weights = (y_true * 50.0) + 1.0
#     bce = K.binary_crossentropy(y_true, y_pred)
#     weighted_bce = K.mean(bce * weights)
#     return weighted_bce


# def jacard_val(y_true, y_pred):
#     """
#     Intersection-Over-Union (IoU), also known as the Jaccard loss
#     """
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
#     jacard_similarity = intersection / union
#     return 1 - jacard_similarity


# def unet3p_hybrid_loss():
#     def loss(y_true, y_pred):
#         """
#         Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
#         Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
#         which is able to capture both large-scale and fine structures with clear boundaries.
#         """
#         WCE = weighted_bce(y_true, y_pred)
#         FTV = ftv_val(y_true, y_pred)
#         SSIM = ssim_val(y_true, y_pred)
#         JAC = jacard_val(y_true, y_pred)
#         return 0.6 * FTV + 0.1 * SSIM + 0.1 * JAC + 0.2 * WCE

#     return loss
