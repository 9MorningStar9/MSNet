import keras.backend as K
import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, concatenate
from keras.layers import Layer
from keras.models import Model
from keras.optimizers import Adam


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


def MRCNN256(pretrained_weights=None, input_size=(256, 256, 8), classNum=7, learning_rate=1e-4):
    channel = 8
    inputs = Input(input_size)  # 256*256  7
    Conv0 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(inputs)
    # 第一层
    Pool1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(Conv0)  # 128*128
    Conv1 = Conv2D(channel * 2, (3, 3), padding='same', activation='relu')(Pool1)  # 128*128  16
    Conv1 = BatchNormalization()(Conv1)
    Conv1 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv1)

    up1 = MaxUnPooling2D()([Conv1, mask_1])  # 256*256  16

    # 第二层
    Pool2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(Conv1)  # 64*64
    Conv2 = Conv2D(channel * 2, (3, 3), padding='same', activation='relu')(Pool2)  # 64*64  16
    Conv2 = BatchNormalization()(Conv2)
    Conv2 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv2)

    up2 = MaxUnPooling2D(size=(4, 4))([Conv2, mask_2])  # 256*256  16

    # 第三层
    Pool3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(Conv2)  # 32*32
    Conv3 = Conv2D(channel * 2, (3, 3), padding='same', activation='relu')(Pool3)  # 32*32  16
    Conv3 = BatchNormalization()(Conv3)
    Conv3 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv3)

    up3 = MaxUnPooling2D(size=(8, 8))([Conv3, mask_3])  # 256*256  32

    # 第四层
    Pool4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(Conv3)  # 16*16
    Conv4 = Conv2D(channel * 2, (3, 3), padding='same', activation='relu')(Pool4)  # 16*16  16
    Conv4 = BatchNormalization()(Conv4)
    Conv4 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv4)

    up4 = MaxUnPooling2D(size=(16, 16))([Conv4, mask_4])  # 256*256  16

    # 输出层

    Conv_full1 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv0)  # 256*256   16
    Conv_full1 = BatchNormalization()(Conv_full1)
    Concat1_1 = concatenate([Conv_full1, up1])  # 256*256  32
    Conv_full1 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Concat1_1)

    Conv_full2 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Conv_full1)  # 256*256   16
    Conv_full2 = BatchNormalization()(Conv_full2)
    Concat2_2 = concatenate([Conv_full2, up1, up2])  # 256*256  48
    Conv_full2 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(Concat2_2)

    Conv_full3 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(Conv_full2)  # 256*256  32
    Conv_full3 = BatchNormalization()(Conv_full3)
    Concat3_3 = concatenate([Conv_full3, up1, up2, up3])  # 256*256  80
    Conv_full3 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(Concat3_3)

    Conv_full4 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(Conv_full3)  # 256*256  32
    Conv_full4 = BatchNormalization()(Conv_full4)
    Concat4_4 = concatenate([Conv_full4, up1, up2, up3, up4])  # 256*256  96
    Conv_full4 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(Concat4_4)

    # Conv5 = Conv2D(classNum, (1, 1), activation='relu', padding='same')(Conv_full4)
    # Conv5 = Conv2D(classNum, (3, 3), activation='relu', padding='same')(Conv5)
    Conv5 = Conv2D(classNum, (3, 3), activation='relu', padding='same')(Conv_full4)
    Conv5 = Conv2D(classNum, (1, 1), activation='softmax')(Conv5)
    model = Model(inputs=inputs, outputs=Conv5)
    loss_weights = {0, 1, 1, 1, 1, 1, 1}
    # {0, 1, 1, 1, 1, 1} {0, loss_crop, loss_built,  loss_wetland, loss_grass, loss_water, loss_mangrove}
    loss_weights = tf.convert_to_tensor(list(loss_weights), dtype=tf.float32)  # 转换为float32
    loss_weights = loss_weights.numpy()
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']
                  , loss_weights=loss_weights)

    #  如果有预训练的权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# model = MRCNN256(pretrained_weights=None, input_size=(256, 256, 8), classNum=7, learning_rate=1e-4)
# print(model.summary())
