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

batch_size = 3
epochs = 25

#### Train ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
checkpoint = ModelCheckpoint('output/icnet_large_full_{epoch:03d}_{categorical_accuracy:.3f}.h5', mode='max')
tensorboard = TensorBoard(batch_size=batch_size, log_dir="./logs/ICNet/large_full/{}/".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())))
lr_decay = LearningRateScheduler(PolyDecay(0.01, 0.9, epochs).scheduler)

# Generators
train_generator = utils.generator(df=utils.load_data(), batch_size=batch_size,
                                  resize_shape=(1024, 512), n_classes=34, training=False, crop_shape=(1024, 512))

# image, label = next(train_generator)
# img = np.array(image[0,:,:,:], dtype=np.float32)
# label = cv2.resize(label[0], (1024, 512))
# print label.shape
# print img.shape
# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.imshow(img / 255)
# plt.subplot(1, 2, 2)
# plt.imshow(label[:, :, 7], cmap='gray') # utils.convert_class_to_rgb(utils._filter_labels(label))
# plt.show()

# Optimizer
optim = optimizers.SGD(lr=0.01, momentum=0.9)

# Model
net = ICNet(width=1024, height=512, n_classes=34, weight_path="./output/icnet_large_full_020_0.800.h5", training=False)
# print(net.model.summary())
# Training

net.model.compile(optim, 'categorical_crossentropy', metrics=['categorical_accuracy'])
net.model.fit_generator(generator=train_generator, steps_per_epoch=1000, epochs=epochs,
                        callbacks=[checkpoint, tensorboard, lr_decay], shuffle=True,
                        max_queue_size=5, use_multiprocessing=True, workers=12, initial_epoch=20)
