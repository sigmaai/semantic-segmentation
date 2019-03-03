#!/usr/bin/python

from time import gmtime, strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler

import utils
from utils import PolyDecay
from models.icnet import ICNet
import configs

## Parameters:

batch_size = 6
epochs = 25
model_type = "large_full_2"

#### Train ####

# Callbacks
checkpoint = ModelCheckpoint('output/icnet_' + model_type + '_{epoch:03d}_{categorical_accuracy:.3f}.h5', mode='max')
tensorboard = TensorBoard(batch_size=batch_size,
                          log_dir="./logs/ICNet/" + model_type + "/{}/".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())))
lr_decay = LearningRateScheduler(PolyDecay(0.01, 0.9, epochs).scheduler)

# Generators
train_generator = utils.generator(df=utils.load_train_data(),
                                  batch_size=batch_size,
                                  resize_shape=(configs.img_width, configs.img_height),
                                  crop_shape=(configs.img_width, configs.img_height),
                                  n_classes=34,
                                  training=True)

val_generator = utils.generator(df=utils.load_val_data(configs.val_label_path), batch_size=1,
                                resize_shape=(configs.img_width, configs.img_height),
                                crop_shape=(configs.img_width, configs.img_height),
                                n_classes=34,
                                training=False)


# Optimizer
optim = optimizers.SGD(lr=0.01, momentum=0.9)

# Model
net = ICNet(width=configs.img_width, height=configs.img_height, n_classes=34,
            weight_path='output/icnet_large_full_2_009_0.787.h5', training=False)

# Training
net.model.compile(optim, 'categorical_crossentropy', metrics=['categorical_accuracy'])
net.model.fit_generator(# training
                        generator=train_generator, steps_per_epoch=1500, epochs=epochs,
                        # validation
                        validation_data=val_generator, validation_steps=500,
                        # callbacks & others
                        callbacks=[checkpoint, tensorboard, lr_decay], shuffle=True,
                        max_queue_size=5, use_multiprocessing=True, workers=12, initial_epoch=10)
