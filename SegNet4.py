from keras import Model, layers
from keras.layers import Input, BatchNormalization, Activation, Conv2D
from keras.optimizers import Adam
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self.padding,
            "pool_size": self.pool_size,
            "strides": self.strides,
        })
        return config


class MaxUnPooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnPooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None, **kwargs):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0],
                                input_shape[1] * self.size[0],
                                input_shape[2] * self.size[1],
                                input_shape[3])
                # self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size,
        })
        return config


def SegNet(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001):
    inputs = Input(input_size)
    # Block 1
    channel = 8
    x = layers.Conv2D(channel, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(channel, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x, mask_1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(channel * 2, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(channel * 2, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x, mask_2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(channel * 4, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(channel * 4, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(channel * 4, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x, mask_3 = MaxPoolingWithArgmax2D(name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x, mask_4 = MaxPoolingWithArgmax2D(name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(channel * 8, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x, mask_5 = MaxPoolingWithArgmax2D(name='block5_pool')(x)

    # decoder
    UnPool_1 = MaxUnPooling2D()([x, mask_5])
    y = Conv2D(channel * 8, (3, 3), padding="same")(UnPool_1)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 8, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 8, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    UnPool_2 = MaxUnPooling2D()([y, mask_4])
    y = Conv2D(channel * 8, (3, 3), padding="same")(UnPool_2)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 8, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 4, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    UnPool_3 = MaxUnPooling2D()([y, mask_3])
    y = Conv2D(channel * 4, (3, 3), padding="same")(UnPool_3)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 4, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel * 2, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    UnPool_4 = MaxUnPooling2D()([y, mask_2])
    y = Conv2D(channel * 2, (3, 3), padding="same")(UnPool_4)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(channel, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    UnPool_5 = MaxUnPooling2D()([y, mask_1])
    y = Conv2D(channel, (3, 3), padding="same")(UnPool_5)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(classNum, (1, 1), padding="same", activation='softmax')(y)
    model = Model(inputs=inputs, outputs=y)
    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    #  如果有预训练的权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# md = SegNet(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001)
# print(md.summary())
