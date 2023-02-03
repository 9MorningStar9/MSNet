from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


def FCN(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001):
    inputs = Input(input_size)

    conv1 = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    score_pool3 = Conv2D(filters=classNum, kernel_size=(3, 3), padding='same', activation='relu')(pool3)

    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    score_pool4 = Conv2D(filters=classNum, kernel_size=(3, 3), padding='same', activation='relu')(pool4)

    conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', )(pool4)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', )(conv5)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', )(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 1×1卷积部分，加入了Dropout层以免过拟合
    fc6 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu', )(pool5)
    fc6 = Dropout(0.3)(fc6)

    fc7 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', )(fc6)
    fc7 = Dropout(0.3)(fc7)
    # 下面的代码为跳层连接结构
    score_fr = Conv2D(filters=classNum, kernel_size=(1, 1), padding='same', activation='relu')(fc7)
    score2 = Conv2DTranspose(filters=classNum, kernel_size=(2, 2), strides=(2, 2), padding="valid",
                             activation=None)(score_fr)
    add1 = add(inputs=[score2, score_pool4], name="add_1")
    score4 = Conv2DTranspose(filters=classNum, kernel_size=(2, 2), strides=(2, 2), padding="valid",
                             activation=None)(add1)
    add2 = add(inputs=[score4, score_pool3], name="add_2")
    UpSample = Conv2DTranspose(filters=classNum, kernel_size=(8, 8), strides=(8, 8), padding="valid",
                             activation=None)(add2)

    outputs = Conv2D(2, 1, activation='softmax')(UpSample)
    # 因softmax的特性，跳层连接部分的卷积层都有nClasses个卷积核，以保证softmax的运行
    # 基于Model方法构建模型
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    #  如果有预训练的权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

#  channel = 8
#  conv_1 = Conv2D(channel, kernel_size=3, activation="relu", padding="same")(inputs)
#  conv_1 = BatchNormalization()(
#      Conv2D(channel, kernel_size=3, activation="relu", padding="same")(conv_1))
#  max_pool_1 = MaxPool2D(pool_size=2, strides=2)(conv_1)
#
#  conv_2 = Conv2D(channel * 2, kernel_size=3, activation="relu", padding="same")(max_pool_1)
#  conv_2 = BatchNormalization()(
#      Conv2D(channel * 2, kernel_size=3, activation="relu", padding="same")(conv_2))
#  max_pool_2 = MaxPool2D(pool_size=2, strides=2)(conv_2)
#
#  conv_3 = Conv2D(channel * 4, kernel_size=3, activation="relu", padding="same")(max_pool_2)
#  conv_3 = Conv2D(channel * 4, kernel_size=3, activation="relu", padding="same")(conv_3)
#  conv_3 = Conv2D(channel * 4, kernel_size=3, activation="relu", padding="same")(conv_3)
#  conv_3 = BatchNormalization()(
#      Conv2D(channel * 4, kernel_size=3, activation="relu", padding="same")(conv_3))
#  max_pool_3 = MaxPool2D(pool_size=2, strides=2)(conv_3)
#
#  conv_4 = Conv2D(channel * 8, kernel_size=3, activation="relu", padding="same")(max_pool_3)
#  conv_4 = Conv2D(channel * 8, kernel_size=3, activation="relu", padding="same")(conv_4)
#  conv_4 = Conv2D(channel * 8, kernel_size=3, activation="relu", padding="same")(conv_4)
#  conv_4 = BatchNormalization()(
#      Conv2D(channel * 8, kernel_size=3, activation="relu", padding="same")(conv_4))
#  max_pool_4 = MaxPool2D(pool_size=2, strides=2)(conv_4)
#
#  conv_5 = Conv2D(channel * 16, kernel_size=3, activation="relu", padding="same")(max_pool_4)
#  conv_5 = Conv2D(channel * 16, kernel_size=3, activation="relu", padding="same")(conv_5)
#  conv_5 = Conv2D(channel * 16, kernel_size=3, activation="relu", padding="same")(conv_5)
#  conv_5 = BatchNormalization()(
#      Conv2D(channel * 16, kernel_size=3, activation="relu", padding="same")(conv_5))
#
#  ups-amping_6 = UpSampling2D(size=16)(conv_5)
#
#  conv = Conv2D(2, (3, 3), activation='relu', padding='same')(ups-amping_6)
#  outputs = Conv2D(classNum, (1, 1), activation='softmax')(conv)


# md = FCN(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001)
# print(md.summary())
