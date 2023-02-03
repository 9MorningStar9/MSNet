from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, SpatialDropout2D, concatenate, UpSampling2D
from keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001):
    inputs = Input(input_size)
    #  2D卷积层
    channel = 8
    conv1 = Conv2D(channel, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(trainable=False)(conv1)
    conv1 = Conv2D(channel, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(trainable=False)(conv2)
    conv2 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(trainable=False)(conv3)
    conv3 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(trainable=False)(conv4)
    conv4 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(channel * 16, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization(trainable=False)(conv5)
    conv5 = Conv2D(channel * 16, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = SpatialDropout2D(0.35)(up6)
    conv6 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(channel * 8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = SpatialDropout2D(0.35)(up7)
    conv7 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(channel * 4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = SpatialDropout2D(0.35)(up8)
    conv8 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(channel * 2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = SpatialDropout2D(0.35)(up9)
    conv9 = Conv2D(channel, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(channel, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = Conv2D(classNum, (1, 1), activation='softmax')(conv10)
    model = Model(inputs=inputs, outputs=conv10)

    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    #  如果有预训练的权重
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# md = unet(pretrained_weights=None, input_size=(256, 256, 7), classNum=2, learning_rate=0.001)
# print(md.summary())
