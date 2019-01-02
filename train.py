#
# Training script for semantic segmentation
# Lyft Udacity Perception challenge
# (c) Yongyang Nie
#


import models.enet_naive_upsampling.model as enet
from models.icnet import ICNet
from keras.callbacks import TensorBoard
from keras import optimizers
import matplotlib.pyplot as plt
import configs
import utils
import numpy as np
import pandas
import os
import datetime


def train(model, epochs, steps_epch, train_gen, save_path, test_result=False):

    tensorboard = TensorBoard(log_dir=("logs/" + save_path + "/{}".format(datetime.datetime.now())))

    optimizer = optimizers.SGD(lr=0.01, momentum=0.9)

    # Training
    model.compile(optimizer, 'categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit_generator(generator=train_gen, steps_per_epoch=steps_epch, epochs=epochs, verbose=1, callbacks=[tensorboard])
    model.save(save_path + ".h5")

    if test_result:

        # Plotting generator output
        images, targets = next(train_generator)

        for i in range(len(images)):
            im_gt = np.array(targets[i], dtype=np.uint8)
            im_prediction = model.predict(np.array(images[i], dtype=np.uint8))[0]
            plt.subplot(1, 3, 1)
            plt.imshow(np.array(images[i], dtype=np.uint8))
            plt.subplot(1, 3, 2)
            plt.imshow(im_gt[:, :, 0])
            plt.subplot(1, 3, 3)
            plt.imshow(im_prediction[:, :, 0])
            plt.show()


if __name__ == "__main__":

    # model = enet.build(len(utils.labels), configs.img_height, configs.img_width)
    # print(model.summary())

    model = ICNet(width=512, height=512, n_classes=len(utils.labels))
    print(model.model.summary())
    exit(0)
    df = load_data()

    train_generator = utils.train_generator(df, 1)

    # Plotting generator output
    # images, targets = next(train_generator)
    #
    # for i in range(len(images)):
    #     im_gt = np.array(targets[i])
    #     im_prediction = model.predict(np.array([images[i]]))[0]
    #     print im_prediction.shape
    #     print im_prediction[:, :, 0]
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(np.array(images[i]))
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(utils.convert_class_to_rgb(im_gt))
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(utils.convert_class_to_rgb(im_prediction))
    #     plt.show()

    # model.load_weights("./enet_v7.h5")
    train(model.model, epochs=2, train_gen=train_generator, steps_epch=1000, save_path="./icnet_1")
