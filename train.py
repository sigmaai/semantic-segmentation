#
# Training script for semantic segmentation
# Lyft Udacity Perception challenge
# (c) Yongyang Nie
#


import models.enet_naive_upsampling.model as enet
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import configs
import utils
import numpy as np
import pandas
import os
import datetime


def train(model, epochs, steps_epch, train_gen, save_path, test_result=False):

    tensorboard = TensorBoard(log_dir=("logs/" + save_path + "/{}".format(datetime.datetime.now())))

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


def load_data():

    labels = pandas.read_csv(configs.data_path).values
    df = []
    count = 0
    for row in labels:
        if os.path.isfile(row[0]) and os.path.isfile(row[1]):
            count = count + 1
            df.append(row)

    print("data processing finished")
    print("data frame size: " + str(count))

    return df


if __name__ == "__main__":

    model = enet.build(len(utils.labels), configs.img_height, configs.img_width)
    print(model.summary())

    df = load_data()

    train_generator = utils.train_generator(df, 4)

    model.load_weights("./enet_v3.h5")
    train(model, epochs=5, train_gen=train_generator, steps_epch=200, save_path="./enet_v4")
