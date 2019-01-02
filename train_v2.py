#!/usr/bin/python

from time import gmtime, strftime

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
epochs = 5

#### Train ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Callbacks
# checkpoint = ModelCheckpoint('output/weights.{epoch:03d}-{val_conv6_cls_categorical_accuracy:.3f}.h5', monitor='val_conv6_cls_categorical_accuracy', mode='max')
tensorboard = TensorBoard(batch_size=batch_size, log_dir="ICNet/{}/".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())))
lr_decay = LearningRateScheduler(PolyDecay(0.01, 0.9, epochs).scheduler)

# Generators
train_generator = utils.train_generator(df=utils.load_data(), batch_size=batch_size, crop_shape=(512, 512), n_classes=34)
X, [Y1, Y2, Y3] = next(train_generator)

# Optimizer
optim = optimizers.SGD(lr=0.01, momentum=0.9)

# Model
net = ICNet(width=512, height=512, n_classes=13, weight_path=None, training=True)

# Training
net.model.compile(optim, 'categorical_crossentropy', loss_weights=[1.0, 0.4, 0.16], metrics=['categorical_accuracy'])
net.model.fit_generator(generator=train_generator, steps_per_epoch=1000, epochs=epochs, callbacks=[tensorboard, lr_decay],
                        use_multiprocessing=False, shuffle=True, max_queue_size=10)

net.model.save("icnet-v1.h5")
