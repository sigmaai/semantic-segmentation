#!/usr/bin/python

from time import gmtime, strftime
import matplotlib.pyplot as plt
import cv2
import numpy as np

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

import utils
from utils import PolyDecay
from models.icnet import ICNet
import configs

## Parameters:

batch_size = 5
epochs = 20

#### Train ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
checkpoint = ModelCheckpoint('output/weights.{epoch:03d}-{val_conv6_cls_categorical_accuracy:.3f}.h5', monitor='val_conv6_cls_categorical_accuracy', mode='max')
tensorboard = TensorBoard(batch_size=batch_size, log_dir="./logs/ICNet/{}/".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())))
lr_decay = LearningRateScheduler(PolyDecay(0.01, 0.9, epochs).scheduler)

# Generators
train_generator = utils.generator(df=utils.load_data(), batch_size=batch_size,
                                  resize_shape=(512, 512), n_classes=34, training=False, crop_shape=(512, 512))

# image, label = next(train_generator)
# img = np.array(image[0,:,:,:], dtype=np.float32)
# label = cv2.resize(label[0], (512, 512))
# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.imshow(img / 255)
# plt.subplot(1, 2, 2)
# plt.imshow(utils.convert_class_to_rgb(label))
# plt.show()

# Optimizer
optim = optimizers.SGD(lr=0.01, momentum=0.9)

# Model
net = ICNet(width=512, height=512, n_classes=13, weight_path=None, training=False)
print(net.model.summary())
# Training

net.model.load_weights("icnet3-v9.h5")
net.model.compile(optim, 'categorical_crossentropy', metrics=['categorical_accuracy'])
net.model.fit_generator(generator=train_generator, steps_per_epoch=1000, epochs=epochs, callbacks=[tensorboard, lr_decay],
                        shuffle=True, max_queue_size=5)

net.model.save("icnet3-v10.h5")
