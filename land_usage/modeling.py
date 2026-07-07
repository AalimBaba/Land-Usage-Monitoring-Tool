from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from .config import MODEL_INPUT_SIZE, NUM_CLASSES


def build_unet(
    input_shape: tuple[int, int, int] = (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3),
    num_classes: int = NUM_CLASSES,
    base_filters: int = 32,
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    bridge = conv_block(p3, base_filters * 8)

    u4 = up_block(bridge, c3, base_filters * 4)
    u5 = up_block(u4, c2, base_filters * 2)
    u6 = up_block(u5, c1, base_filters)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(u6)
    return models.Model(inputs, outputs, name="land_usage_unet")


def conv_block(inputs, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def up_block(inputs, skip, filters: int):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = layers.Concatenate()([x, skip])
    return conv_block(x, filters)


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def sparse_focal_loss(y_true, y_pred, gamma: float = 2.0):
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=NUM_CLASSES)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    focal_weight = tf.pow(1.0 - y_pred, gamma)
    return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))


def combined_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(ce) + dice_loss(y_true, y_pred) + 0.25 * sparse_focal_loss(y_true, y_pred)

