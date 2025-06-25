# Siamese Model from: "Fast Video Object Segmentation by Reference-Guided Mask Propagation"
#  https://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf

import numpy as np
import tensorflow as tf
import keras
from keras import Model
from keras.layers import (
    ZeroPadding2D,
    Softmax,
    Add,
    UpSampling2D,
    Conv2D,
    Concatenate,
    ReLU,
    Input,
    BatchNormalization,
    LeakyReLU,
)

from PIL import Image
from keras.utils import plot_model


def refineblock(skip_connection, input, refineName):
    conv3x3 = Conv2D(
        filters=256, kernel_size=(3, 3), padding="same", name=refineName + "conv3x3"
    )(skip_connection)
    res_block = resblock(conv3x3, (3, 3), 256, refineName + "_1")
    upsample = UpSampling2D(name=refineName + "upSample")(input)
    out = Add(name=refineName + "refAdd")([upsample, res_block])
    return resblock(out, (3, 3), 256, refineName + "_2")
    # return upsample


def resblock(x, kernelsize, filters, resblockname):
    fx = Conv2D(
        filters,
        kernelsize,
        activation="relu",
        padding="same",
        name=resblockname + "Conv2Dfx",
    )(x)
    fx = Conv2D(filters, kernelsize, padding="same", name=resblockname + "Conv2Dfx2")(
        fx
    )
    out = Add(name=resblockname + "resAdd")([x, fx])
    out = ReLU(name=resblockname + "resRelu")(out)
    return out


def GlobalConvBlock(input):
    conv1x7_up = Conv2D(
        filters=256, kernel_size=(1, 7), padding="same", name="conv1x7_up"
    )(input)
    conv7x1_up = Conv2D(
        filters=256, kernel_size=(7, 1), padding="same", name="conv7x1_up"
    )(conv1x7_up)

    conv7x1_down = Conv2D(
        filters=256, kernel_size=(7, 1), padding="same", name="conv7x1_down"
    )(input)
    conv1x7_down = Conv2D(
        filters=256, kernel_size=(1, 7), padding="same", name="conv1x7_down"
    )(conv7x1_down)

    summation_layer = Add(name="Add")([conv7x1_up, conv1x7_down])
    return resblock(summation_layer, (3, 3), 256, "global")


input_shape = (480, 854, 4)


# Target Stream  = Q
inputlayer_Q = Input(shape=input_shape, name="inputlayer_Q")
# Refrence Stream = M
inputlayer_M = Input(shape=input_shape, name="inputlayer_M")


convlayer_Q = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(inputlayer_Q)
convlayer_M = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(inputlayer_M)

# out = main_model([convlayer_Q,convlayer_M])

# new_model = Model(inputs=[inputlayer_Q, inputlayer_M],outputs=out, name ="new_model" )


model_Q = keras.applications.resnet50.ResNet50(
    input_shape=(convlayer_Q.shape[1], convlayer_Q.shape[2], convlayer_Q.shape[3]),
    include_top=False,
    weights="imagenet",
)
model_Q._name = "resnet50_Q"

model_M = keras.applications.resnet50.ResNet50(
    input_shape=(convlayer_M.shape[1], convlayer_M.shape[2], convlayer_M.shape[3]),
    include_top=False,
    weights="imagenet",
)
model_M._name = "resnet50_M"

for model in [model_Q, model_M]:
    for layer in model.layers:
        old_name = layer.name
        layer._name = f"{model.name}_{old_name}"
        print(layer._name)


encoder_Q = Model(inputs=model_Q.inputs, outputs=model_Q.output, name="encoder_Q")
encoder_M = Model(inputs=model_M.inputs, outputs=model_M.output, name="encoder_M")


concatenate = Concatenate(axis=0, name="Concatenate")(
    [encoder_Q.output, encoder_M.output]
)
global_layer = GlobalConvBlock(concatenate)

res2_skip = encoder_Q.layers[38].output
res2_skip = ZeroPadding2D(padding=(0, 1), data_format=None)(res2_skip)
res3_skip = encoder_Q.layers[80].output
res3_skip = ZeroPadding2D(padding=((0, 0), (0, 1)), data_format=None)(res3_skip)
res4_skip = encoder_Q.layers[142].output


ref1_16 = refineblock(res4_skip, global_layer, "ref1_16")
ref1_8 = refineblock(res3_skip, ref1_16, "ref1_8")
ref1_4 = refineblock(res2_skip, ref1_8, "ref1_4")
outconv = Conv2D(filters=2, kernel_size=(3, 3))(ref1_4)
outconv1 = ZeroPadding2D(padding=((1, 1), (0, 0)), data_format=None)(outconv)
output = Softmax()(outconv1)

main_model = Model(
    inputs=[encoder_Q.inputs, encoder_M.inputs], outputs=output, name="main model"
)
