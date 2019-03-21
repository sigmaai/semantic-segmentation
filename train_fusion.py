#!/usr/bin/python

from time import gmtime, strftime

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import plot_model

import utils
from utils import PolyDecay
from models.icnet_fusion import ICNet
import configs

# ==========
# Parameters
# ==========
batch_size = 3
epochs = 10
init_epoch = 5
model_type = "cross_fusion_gl"

#### Train ####

# Callbacks
checkpoint = ModelCheckpoint('output/icnet_' + model_type + '_{epoch:03d}_{conv6_cls_categorical_accuracy:.3f}.h5', mode='max')
tensorboard = TensorBoard(batch_size=batch_size,
                          log_dir="./logs/ICNet/" + model_type + "/{}/".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())))
lr_decay = LearningRateScheduler(PolyDecay(0.01, 0.9, epochs).scheduler)

# ==========
# Generators
# ==========
if model_type == "early_fusion":
    train_generator = utils.early_fusion_generator(df=utils.load_train_data(configs.label_depth_color_path),
                                                   batch_size=batch_size,
                                                   resize_shape=(configs.img_width, configs.img_height),
                                                   crop_shape=(configs.img_width, configs.img_height),
                                                   n_classes=34,
                                                   training=True)

    val_generator = utils.early_fusion_generator(df=utils.load_val_data(configs.val_depth_color_path), batch_size=1,
                                                 resize_shape=(configs.img_width, configs.img_height),
                                                 crop_shape=(configs.img_width, configs.img_height),
                                                 n_classes=34,
                                                 training=False)
elif "cross_fusion" in model_type:
    train_generator = utils.fusion_generator(df=utils.load_train_data(configs.label_depth_color_path),
                                                 batch_size=batch_size,
                                                 resize_shape=(configs.img_width, configs.img_height),
                                                 n_classes=34,
                                                 training=True)

    val_generator = utils.fusion_generator(df=utils.load_val_data(configs.val_depth_color_path), batch_size=1,
                                               resize_shape=(configs.img_width, configs.img_height),
                                               n_classes=34,
                                               training=False)
else:
    raise ValueError("Model type not found")

# Optimizer
optim = optimizers.SGD(lr=0.01, momentum=0.9)

# Model
net = ICNet(width=configs.img_width, height=configs.img_height, n_classes=34, depth=6, mode=model_type, training=True,
            weight_path='output/icnet_' + model_type + '_004_0.776.h5')

print(net.model.summary())

plot_model(net.model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Training
net.model.compile(optim, 'categorical_crossentropy', metrics=['categorical_accuracy'])
net.model.fit_generator(generator=train_generator, steps_per_epoch=1500, epochs=epochs,
                        validation_data=val_generator, validation_steps=800,
                        callbacks=[checkpoint, tensorboard, lr_decay], shuffle=True,
                        max_queue_size=5, use_multiprocessing=True, workers=12, initial_epoch=init_epoch)
