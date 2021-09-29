# from keras import backend as keras
from keras.layers import *
from keras.layers import Dense
from keras import layers as ly
from keras.models import *
from keras.optimizers import *
import tensorflow.compat.v1 as tf
import cv2
import gdal
import matplotlib.pyplot as plt

def Conv_bn_relu(num_filters,
                 kernel_size,
                 batchnorm=True,
                 strides=(1, 1),
                 padding='same'):

    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer

def resize256(inputimg):
    yy = tf.image.resize(inputimg,(256,256))
    return yy

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list

def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate

def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)


    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)


    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate

def attention_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = down_layer
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    # up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate

def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    print(inter_channel,'inter_channel')
    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer

def rec_res_block_d5(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    # for j in range(2):
    #     for i in range(2):
    #         if i == 0:
    #             layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
    #             if batch_normalization:
    #                 layer1 = BatchNormalization()(layer1)
    #             layer1 = Activation('relu')(layer1)
    #         layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(add([layer1, layer]))
    #         if batch_normalization:
    #             layer1 = BatchNormalization()(layer1)
    #         layer1 = Activation('relu')(layer1)
    #     layer = layer1
    layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
    if batch_normalization:
        layer1 = BatchNormalization()(layer1)
    temp_layer = add([layer, layer1])
    layer2 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer1)
    if batch_normalization:
        layer2 = BatchNormalization()(layer2)
    temp_layer = add([layer2, temp_layer])
    layer3 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer2)
    if batch_normalization:
        layer3 = BatchNormalization()(layer3)
    temp_layer = add([layer3, temp_layer])
    layer4 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer3)
    if batch_normalization:
        layer4 = BatchNormalization()(layer4)
    temp_layer = add([layer4, temp_layer])
    out_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(temp_layer)
    out_layer = BatchNormalization()(out_layer)
    out_layer = Activation('relu')(out_layer)
    out_layer = add([out_layer, skip_layer])

    return out_layer

def rec_res_block_d2(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    # for j in range(2):
    #     for i in range(2):
    #         if i == 0:
    #             layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
    #             if batch_normalization:
    #                 layer1 = BatchNormalization()(layer1)
    #             layer1 = Activation('relu')(layer1)
    #         layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(add([layer1, layer]))
    #         if batch_normalization:
    #             layer1 = BatchNormalization()(layer1)
    #         layer1 = Activation('relu')(layer1)
    #     layer = layer1
    layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
    if batch_normalization:
        layer1 = BatchNormalization()(layer1)
    temp_layer = add([layer, layer1])
    layer2 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer1)
    if batch_normalization:
        layer2 = BatchNormalization()(layer2)
    temp_layer = add([layer2, temp_layer])
    out_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(temp_layer)
    out_layer = BatchNormalization()(out_layer)
    out_layer = Activation('relu')(out_layer)
    out_layer = add([out_layer, skip_layer])

    return out_layer

def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_first'):
    activation_name = 'tanh'#'relu'
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation(activation_name)(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation(activation_name)(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])

    return out_layer
def rec_res_block_d3(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

def res2net_block(num_filters, slice_num,input_tensor):
    short_cut = input_tensor
    x = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(input_tensor)
    slice_list = slice_layer(x, slice_num, x.shape[-1])
    side = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
    z = concatenate([slice_list[0], side])   # for one and second stage
    for i in range(2, len(slice_list)):
        y = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
        side = y
        z = concatenate([z, y])
    z = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(z)
    out = concatenate([z, short_cut])
    return out

def resLayer1(input_tensor):
    conv11 = res2net_block(64, 4,input_tensor) #print(shape)
    conv12 = res2net_block(64, 4,conv11)
    conv13 = res2net_block(256, 4, conv12)
    return conv13
def resLayer2(input_tensor):
    conv21 = res2net_block(64, 4, input_tensor)
    conv22 = res2net_block(64, 4, conv21)
    conv23 = res2net_block(256, 4, conv22)
    conv24 = res2net_block(256, 4, conv23)
    return conv24
def resLayer3(input_tensor):
    conv31 = res2net_block(14, 4, input_tensor)
    conv32 = res2net_block(14, 4, conv31)
    conv33 = res2net_block(14, 4, conv32)
    conv34 = res2net_block(14, 4, conv33)
    conv35 = res2net_block(14, 4, conv34)
    conv36 = res2net_block(14, 4, conv35)
    return conv36
def resLayer4(input_tensor):
    conv41 = res2net_block(14, 4, input_tensor)
    conv42 = res2net_block(14, 4, conv41)
    conv43 = res2net_block(14, 4, conv42)
    return conv43
#多分类
def ccy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)
def unet(pretrained_weights= 'unet_membrane.hdf5', input_size=(256, 256, 50)):#'unet_membrane.hdf5'
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #其中最后一行因为所做的是x分类，所以第一个数字是
    conv10 = Conv2D(6, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# Res2net
# def unet(pretrained_weights='unet_membrane.hdf5', input_size=(256, 256, 47)):  # 'unet_membrane.hdf5'
#     inputs = Input(input_size)
#     # print(inputs.shape,'inputs')
#     conv1 = Conv2D(64, 7, activation='relu', padding="same", kernel_initializer='he_normal')(inputs)#128x128
#     pool1 = MaxPooling2D(pool_size=(3, 3),padding="same")(conv1)#256*64x64
#     layer1 = Lambda(resLayer1)(pool1)
#     pool2 = MaxPooling2D(pool_size=(3, 3),padding="same")(layer1)#512*32x32
#     layer2 = Lambda(resLayer2)(pool2)
#     pool3 = MaxPooling2D(pool_size=(3, 3),padding="same")(layer2)#1024*16x16
#     layer3 = Lambda(resLayer3)(pool3)
#     pool4 = MaxPooling2D(pool_size=(3, 3),padding="same")(layer3)#2048*8x8
#     layer4 = Lambda(resLayer4)(pool4)
#     up1 = Conv2D(2048, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(layer4))#2048*16x16
#     cov1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
#     cov1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov1)
#     up2 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(cov1))#2048*32x32
#     cov2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
#     cov2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov2)
#     # connect1 = concatenate([pool2, cov2], axis=3)
#     end = Lambda(resize256)(cov2)
#     # avg5 = GlobalAvgPool2D()(end)
#     end2 = Conv2D(6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(end)
#     # print(Denselayer.shape,'Denselayer')
#     model = Model(input=inputs, output=end2)
#     model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model

#网络2
def att_unet(img_w, img_h, n_label):
    inputs = Input((3, img_w, img_h))
    x = inputs
    data_format = 'channels_first'
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_first')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model

def att_r2_unet(pretrained_weights='unet_membrane.hdf5', input_size=(256, 256, 50)):#'unet_membrane.hdf5'
    inputs = Input(input_size)
    x = inputs
    depth = 4
    depth2 = 4
    features = 64
    data_format = 'channels_last'
    n_label = 6
    skips = []

    x = attention_and_concate(x, inputs, data_format=data_format)

    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)

        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        # x = attention_up_and_concate(x, skips[i], data_format=data_format)
        # x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        # x = concatenate([x, skips[i]])
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    # conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv6 = Conv2D(n_label, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(x)
    # conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv6)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy']) #   'categorical_crossentropy'

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def resnet(pretrained_weights='unet_membrane.hdf5', input_size=(256, 256, 50)):  # 'unet_membrane.hdf5'
    inputs = Input(input_size)

    conv1 = Conv2D(64, 7, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv21)
    merge11 = concatenate([pool1,conv22], axis = 3)
    conv23 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv24 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv23)
    merge12 = concatenate([merge11,conv24], axis = 3)

    conv31 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv32 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv31)
    merge31 = concatenate([merge12,conv32], axis = 3)
    conv33 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge31)
    conv34 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv33)
    merge32 = concatenate([merge31,conv34], axis = 3)

    conv41 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge32)
    conv42 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv41)
    merge41 = concatenate([merge32,conv42], axis = 3)
    conv43 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge41)
    conv44 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv43)
    merge42 = concatenate([merge41,conv44], axis = 3)

    conv51 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge42)
    conv52 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv51)
    merge51 = concatenate([merge42,conv52], axis = 3)
    conv53 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge51)
    conv54 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv53)
    merge52 = concatenate([merge51,conv54], axis = 3)

    pool1 = AvgPool2D(pool_size=(2, 2))(merge52)

    up7 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool1))
    up8 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up7))
    conv10 = Conv2D(6, 1, activation='sigmoid')(up8)
    print(conv10.shape,'conv10')

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# def unettest(pretrained_weights=None, input_size=(256, 256, 47)):  # 'unet_membrane.hdf5'
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(pool1))
#
#     # inputs = Input(input_size)
#     # conv1 = Conv2D(64, 7, activation='relu', padding="same", kernel_initializer='he_normal')(inputs)#128x128
#     # # conv1 = Conv2D(64, 7, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     # # pool1 = MaxPooling2D(pool_size=(3, 3),padding="same")(conv1)#256*64x64
#     # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # conv1 = Conv2D(64, 7, activation='relu', padding="same", kernel_initializer='he_normal',strides=2)(inputs)#128x128
#
#     # pool1 = MaxPooling2D(pool_size=(3, 3),strides=2,padding="same")(conv1)#256*64x64
#     # layer1 = Lambda(resLayer1)(pool1)
#
#
#     conv10 = Conv2D(6, 1, activation='softmax')(up9)
#     # print(Denselayer.shape,'Denselayer')
#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model

# 未改动
# def unet(pretrained_weights=None, input_size=(256, 256, 3)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(input=inputs, output=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model

#res2net


# input_size = (512,512,10)EU
# def unet(pretrained_weights = 'unet_membrane.hdf5',input_size = (512,512,10)):#None or 'unet_membrane.hdf5'
#
#     inputs0 = Input(input_size)
#     inputs00 = MaxPooling2D(pool_size=(2, 2))(inputs0)  # 改动 256x256x64
#     # print(input_size,'input size')
#     conv000 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs00)
#     atrous0_3 = Lambda(atrous645)(conv000)
#     atrous0_2 = Lambda(atrous643)(conv000)
#     atrous0_1 = Lambda(atrous642)(conv000)
#
#     res256 = Lambda(resize256)(inputs00)#3
#     res01 = Lambda(resize128)(inputs00)
#     res02 = Lambda(resize64)(inputs00)
#     res03 = Lambda(resize32)(inputs00)
#     res04 = Lambda(resize16)(inputs00)
#     res05 = Lambda(resize8)(inputs00)
#     # print('merge0:',merge0.get_shape())
#
#     conv0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs00)  # 改动 512x512 跨通道整合
#     conv0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)  # 改动 512x512 跨通道整合
#     conv0_21 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs00)
#     conv0_22 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_21)
#     conv0_3 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs00)
#     merge00 = concatenate([conv0,inputs00,atrous0_1,atrous0_2,atrous0_3,conv0_22,conv0_3], axis=3)
#     out0 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00)
#
#     pool00 = MaxPooling2D(pool_size=(2, 2))(out0)  # 改动 256x256x64
#
#     conv01_11 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)#256x256x128
#     conv01_12 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv01_11)#256x256x128
#     conv01_21 = Conv2D(100, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)
#     conv01_22 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv01_21)
#     conv01_31 = Conv2D(100, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)
#     merge1 = concatenate([conv01_12, conv01_22,conv01_31,res01], axis=3)  # 76
#     out1 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
#
#     pool1 = MaxPooling2D(pool_size=(2, 2))(out1)  # 改动 256x256x64
#
#     conv02_11 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#128x128
#     conv02_12 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv02_11)#128x128
#     conv02_21 = Conv2D(300, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # 128x128
#     conv02_22 = Conv2D(300, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv02_21)  # 128x128
#     conv02_31 = Conv2D(300, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # 128x128
#     merge2 = concatenate([conv02_12,conv02_22,conv02_31,res02], axis=3)#改动 特征数量是两个图层只和     改到这里
#     out2 = Conv2D(900, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
#
#     pool2 = MaxPooling2D(pool_size=(2, 2))(out2)#128x128x131 print("pool1:",pool1.get_shape())
#
#     conv03_11 = Conv2D(900, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv03_12 = Conv2D(900, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv03_11)
#     conv03_21 = Conv2D(900, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64x64
#     conv03_22 = Conv2D(900, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv03_21)#64x64
#     conv03_31 = Conv2D(900, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)  # 64x64
#     merge2 = concatenate([conv03_12,conv03_22,conv03_31,res03], axis=3)#改动 128x128x265
#     out3 = Conv2D(1000, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
#
#     drop3 = Dropout(0.5)(out3)#16x16
#
#     up8 = Lambda(resize128)(drop3)
#     conv81 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#     merge8 = concatenate([merge1,conv81,conv8,res01], axis = 3)
#     # print('merge8:',merge8.shape)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#
#
#     up9 = Lambda(resize256)(conv8)
#     conv91 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     merge9 = concatenate([conv9,conv91,merge00], axis = 3)
#     # print('merge9:',merge9.shape)
#     conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#
#
#     up10 = Lambda(resize512)(conv10)
#     conv101 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
#     conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
#     conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     merge10 = concatenate([conv101,conv10], axis = 3)
#     conv10= Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
#
#     conv11 = Lambda(resize512)(conv10)
#     # merge11 = concatenate([conv11,conv10], axis = 3)
#     conv10= Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
#     conv10= Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     conv10= Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     # conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
#     #其中最后一行因为所做的是x分类，所以第一个数字是
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)
#     # print("conv10:", conv10.get_shape())
#     model = Model(input = inputs0, output = conv10)
#
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#     # print(loss)
#     model.summary()
#
#     if(pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model
# #


# inputshape=(256,256,10)
# def unet(pretrained_weights = 'unet_membrane.hdf5',input_size = (256,256,10)):#None or 'unet_membrane.hdf5'
#
#     inputs0 = Input(input_size)
#     inputs00 = MaxPooling2D(pool_size=(2, 2))(inputs0)  # 改动 256x256x64
#     # print(input_size,'input size')
#     conv000 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs00)
#     atrous0_3 = Lambda(atrous645)(conv000)
#     atrous0_2 = Lambda(atrous643)(conv000)
#     atrous0_1 = Lambda(atrous642)(conv000)
#
#     res256 = Lambda(resize256)(inputs00)#3
#     res01 = Lambda(resize128)(inputs00)
#     res02 = Lambda(resize64)(inputs00)
#     res03 = Lambda(resize32)(inputs00)
#     res04 = Lambda(resize16)(inputs00)
#     res05 = Lambda(resize8)(inputs00)
#     # print('merge0:',merge0.get_shape())
#
#     conv0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs00)  # 改动 512x512 跨通道整合
#     conv0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)  # 改动 512x512 跨通道整合
#     conv0_21 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs00)
#     conv0_22 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_21)
#     conv0_3 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs00)
#     merge00 = concatenate([conv0,inputs00,atrous0_1,atrous0_2,atrous0_3,conv0_22,conv0_3], axis=3)
#     out0 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00)
#
#     pool00 = MaxPooling2D(pool_size=(2, 2))(out0)  # 改动 256x256x64
#
#     conv01_11 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)#256x256x128
#     conv01_12 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv01_11)#256x256x128
#     conv01_21 = Conv2D(100, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)
#     conv01_22 = Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv01_21)
#     conv01_31 = Conv2D(100, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool00)
#     merge1 = concatenate([conv01_12, conv01_22,conv01_31,res02], axis=3)  # 76
#     out1 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
#
#     pool1 = MaxPooling2D(pool_size=(2, 2))(out1)  # 改动 256x256x64
#
#     conv02_11 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)#128x128
#     conv02_12 = Conv2D(300, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv02_11)#128x128
#     conv02_21 = Conv2D(300, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # 128x128
#     conv02_22 = Conv2D(300, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv02_21)  # 128x128
#     conv02_31 = Conv2D(300, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # 128x128
#     merge2 = concatenate([conv02_12,conv02_22,conv02_31,res03], axis=3)#改动 特征数量是两个图层只和     改到这里
#     out2 = Conv2D(900, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
#
#     pool2 = MaxPooling2D(pool_size=(2, 2))(out2)#128x128x131 print("pool1:",pool1.get_shape())
#
#     conv03_11 = Conv2D(900, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv03_12 = Conv2D(900, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv03_11)
#     conv03_21 = Conv2D(900, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)#64x64
#     conv03_22 = Conv2D(900, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv03_21)#64x64
#     conv03_31 = Conv2D(900, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)  # 64x64
#     merge2 = concatenate([conv03_12,conv03_22,conv03_31,res04], axis=3)#改动 128x128x265
#     out3 = Conv2D(1000, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
#
#     drop3 = Dropout(0.5)(out3)#16x16
#
#     up8 = Lambda(resize128)(drop3)
#     conv81 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#     merge8 = concatenate([merge00,conv81,conv8,res01], axis = 3)
#     # print('merge8:',merge8.shape)
#     conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#
#
#     up9 = Lambda(resize256)(conv8)
#     conv91 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
#     conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     merge9 = concatenate([conv9,conv91,inputs0], axis = 3)
#     # print('merge9:',merge9.shape)
#     conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#
#
#     up10 = Lambda(resize512)(conv10)
#     conv101 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
#     conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
#     conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     merge10 = concatenate([conv101,conv10], axis = 3)
#     conv10= Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
#
#     conv11 = Lambda(resize512)(conv10)
#     # merge11 = concatenate([conv11,conv10], axis = 3)
#     conv10= Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
#     conv10= Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     conv10= Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
#     conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
#
#     # print("conv10:", conv10.get_shape())
#     model = Model(input = inputs0, output = conv10)
#
#     model.compile(optimizer = Adam(lr = 1e-7), loss = 'binary_crossentropy', metrics = ['accuracy'])
#     # print(loss)
#     model.summary()
#
#     if(pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model
#Lambda层的作用是自定义层，如果只是对数据进行操作，而网络中的参数没有更新，那么便可以利用Lambda进行自定义层。步骤如下：

##改动U-net
# def unet(pretrained_weights='unet_membrane.hdf5', input_size=(512, 512, 10)):#'unet_membrane.hdf5'
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(input=inputs, output=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model
# #
#
