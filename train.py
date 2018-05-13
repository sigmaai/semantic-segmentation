#
# Training script for semantic segmentation
# Lyft Udacity Perception challenge
# (c) Yongyang Nie
#


import models.enet_naive_upsampling.model as enet
import configs
import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas


def train(model, epochs, train_gen, val_gen, save_path, test_result=False):

    model.fit_generator(train_gen, steps_per_epoch=1000, epochs=epochs, verbose=1)
    model.save(save_path)

    if test_result:

        # Plotting generator output
        images, targets = next(train_generator)

        for i in range(4):
            im = np.array(images[i], dtype=np.uint8)
            im_mask = np.array(targets[i], dtype=np.uint8)
            img = np.array([im], dtype=np.uint8)
            im_prediction = m.predict(img)[0]
            plt.subplot(1, 3, 1)
            plt.imshow(im)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(im_mask[:, :, 0])
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(im_prediction[:, :, 0])
            plt.axis('off')
            plt.show()


if __name__ == "__main__":

    m = enet.build(len(utils.labels), configs.img_height, configs.img_width)
    print(m.summary())

    # TODO: Load the data!!
    label_path = configs.data_path + "extra_labels.csv"
    labels = pandas.read_csv(label_path).values

    # TODO: Fix the train generator code
    train_generator = utils.train_generator(labels, 2)

    train(m, epochs=configs.epochs, train_gen=train_generator, val_gen=None, save_path="./enet_t1.h5")
