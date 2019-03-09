#
# ICNet Keras Implementation
# Paper: https://arxiv.org/abs/1704.08545
# Originally by @aitorzip
# Adapted by @NeilNie
#
# (c) Yongyang Nie, 2018. All Rights Reserved.
# MIT License
#

from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.models import Model
import tensorflow as tf


class ICNet:

    def __init__(self, width, height, n_classes, weight_path=None, depth=6):

        self.width = width
        self.height = height
        self.n_classes = n_classes
        self.weight_path = weight_path

        self.model = self.build_early_fusion(width=self.width, height=self.height, n_classes=self.n_classes, depth=depth)

        if weight_path:
            self.model.load_weights(weight_path)
            print("Model Created \n Weights Loaded. Path: {}".format(weight_path))
        else:
            print("Model Created \n No weight path provided. ")

    def build_early_fusion(self, width, height, n_classes, weights_path=None, depth=6):

        inp_color = Input(shape=(height, width, depth))
        x_color = Lambda(lambda x: (x - 127.5) / 255.0)(inp_color)

        z = self.build_color_one_half(x_color)

        y = self.build_color_one_quarter(z)

        h, w = y.shape[1:3].as_list()
        pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y)
        pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name='conv5_3_pool2')(y)
        pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name='conv5_3_pool3')(y)
        pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
        pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name='conv5_3_pool6')(y)
        pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)

        y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])

        y = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
        y = BatchNormalization(name='conv5_4_k1_bn')(y)

        aux_1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='conv5_4_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
        y = BatchNormalization(name='conv_sub4_bn')(y)
        y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
        y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)
        y = Add(name='sub24_sum')([y, y_])
        y = Activation('relu', name='sub24_sum/relu')(y)

        aux_2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='sub24_sum_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
        y_ = BatchNormalization(name='conv_sub2_bn')(y_)

        y = self.build_color_one(x_color)

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                   name='sub12_sum_interp')(y)

        out = Conv2D(n_classes, 1, activation='softmax', name='conv6_cls')(y)

        model = Model(inputs=inp_color, outputs=out)

        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        return model

    def build_fusion(self, width, height, n_classes, weights_path=None):

        inp_color = Input(shape=(height, width, 3))
        inp_depth = Input(shape=(height, width, 3))
        x_depth = Lambda(lambda x: (x - 127.5) / 255.0)(inp_depth)
        x_color = Lambda(lambda x: (x - 127.5) / 255.0)(inp_color)

        z_color = self.build_color_one_half(x_color)
        y_color = self.build_color_one_half(z_color)

        h, w = y_color.shape[1:3].as_list()
        pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y_color)
        pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name='conv5_3_pool2')(y_color)
        pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name='conv5_3_pool3')(y_color)
        pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
        pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name='conv5_3_pool6')(y_color)
        pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)

        y_color = Add(name='conv5_3_sum')([y_color, pool1, pool2, pool3, pool6])

        y_color = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y_color)
        y_color = BatchNormalization(name='conv5_4_k1_bn')(y_color)

        aux_1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='conv5_4_interp')(y_color)
        y_color = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y_color = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y_color)
        y_color = BatchNormalization(name='conv_sub4_bn')(y_color)
        y_color_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z_color)
        y_color_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_color_)
        y = Add(name='sub24_sum')([y_color, y_color_])
        y_color = Activation('relu', name='sub24_sum/relu')(y_color)

        aux_2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='sub24_sum_interp')(y_color)
        y_color = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_color_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y_color)
        y_color_ = BatchNormalization(name='conv_sub2_bn')(y_color_)

        y_color = self.build_color_one(x_color)

        y_color = Add(name='sub12_sum')([y_color, y_color_])
        y_color = Activation('relu', name='sub12_sum/relu')(y_color)
        y_color = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                   name='sub12_sum_interp')(y_color)

        out = Conv2D(n_classes, 1, activation='softmax', name='conv6_cls')(y_color)

        model = Model(inputs=inp_color, outputs=out)

        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        return model

    @staticmethod
    def build_color_one_half(x_color):

        # (1/2)
        y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
                   name='data_sub2')(x_color)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
        y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
        y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
        y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
        y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
        y = BatchNormalization(name='conv1_3_3x3_bn')(y)
        y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)

        y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)
        y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding1')(y_)
        y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
        y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
        y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
        y = Add(name='conv2_1')([y, y_])
        y_ = Activation('relu', name='conv2_1/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding2')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
        y = BatchNormalization(name='conv2_2_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
        y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
        y = Add(name='conv2_2')([y, y_])
        y_ = Activation('relu', name='conv2_2/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding3')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
        y = BatchNormalization(name='conv2_3_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
        y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
        y = Add(name='conv2_3')([y, y_])
        y_ = Activation('relu', name='conv2_3/relu')(y)

        y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
        y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding4')(y_)
        y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
        y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
        y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
        y = Add(name='conv3_1')([y, y_])
        z = Activation('relu', name='conv3_1/relu')(y)

        return z

    @staticmethod
    def build_color_one_quarter(z):

        # (1/4)
        y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
                    name='conv3_1_sub4')(z)
        y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding5')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
        y = BatchNormalization(name='conv3_2_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
        y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
        y = Add(name='conv3_2')([y, y_])
        y_ = Activation('relu', name='conv3_2/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding6')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
        y = BatchNormalization(name='conv3_3_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
        y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
        y = Add(name='conv3_3')([y, y_])
        y_ = Activation('relu', name='conv3_3/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding7')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
        y = BatchNormalization(name='conv3_4_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
        y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
        y = Add(name='conv3_4')([y, y_])
        y_ = Activation('relu', name='conv3_4/relu')(y)

        y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
        y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
        y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
        y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
        y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
        y = Add(name='conv4_1')([y, y_])
        y_ = Activation('relu', name='conv4_1/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding9')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
        y = BatchNormalization(name='conv4_2_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
        y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
        y = Add(name='conv4_2')([y, y_])
        y_ = Activation('relu', name='conv4_2/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding10')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
        y = BatchNormalization(name='conv4_3_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
        y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
        y = Add(name='conv4_3')([y, y_])
        y_ = Activation('relu', name='conv4_3/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding11')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
        y = BatchNormalization(name='conv4_4_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
        y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
        y = Add(name='conv4_4')([y, y_])
        y_ = Activation('relu', name='conv4_4/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding12')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
        y = BatchNormalization(name='conv4_5_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
        y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
        y = Add(name='conv4_5')([y, y_])
        y_ = Activation('relu', name='conv4_5/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding13')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
        y = BatchNormalization(name='conv4_6_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
        y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
        y = Add(name='conv4_6')([y, y_])
        y = Activation('relu', name='conv4_6/relu')(y)

        y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
        y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
        y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
        y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding14')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
        y = BatchNormalization(name='conv5_1_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
        y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
        y = Add(name='conv5_1')([y, y_])
        y_ = Activation('relu', name='conv5_1/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding15')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
        y = BatchNormalization(name='conv5_2_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
        y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
        y = Add(name='conv5_2')([y, y_])
        y_ = Activation('relu', name='conv5_2/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding16')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
        y = BatchNormalization(name='conv5_3_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
        y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
        y = Add(name='conv5_3')([y, y_])
        y = Activation('relu', name='conv5_3/relu')(y)

        return y

    @staticmethod
    def build_color_one(x_color):

        # (1)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_sub1')(x_color)
        y = BatchNormalization(name='conv1_sub1_bn')(y)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv2_sub1')(y)
        y = BatchNormalization(name='conv2_sub1_bn')(y)
        y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv3_sub1')(y)
        y = BatchNormalization(name='conv3_sub1_bn')(y)
        y = Conv2D(128, 1, name='conv3_sub1_proj')(y)
        y = BatchNormalization(name='conv3_sub1_proj_bn')(y)

        return y



