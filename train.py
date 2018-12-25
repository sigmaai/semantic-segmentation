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
import os


def train(model, epochs, steps_epch, train_gen, save_path, test_result=False):

    model.fit_generator(generator=train_gen, steps_per_epoch=steps_epch, epochs=epochs, verbose=1)
    model.save(save_path)

    if test_result:

        # Plotting generator output
        images, targets = next(train_generator)

        for i in range(4):
            im = np.array(images[i], dtype=np.uint8)
            im_mask = np.array(targets[i], dtype=np.uint8)
            img = np.array([im], dtype=np.uint8)
            im_prediction = model.predict(img)[0]
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

    model = enet.build(len(utils.labels), configs.img_height, configs.img_width)
    print(model.summary())

    label_path = configs.data_path
    labels = pandas.read_csv(label_path).values
    df = []
    count = 0
    for row in labels:
        if os.path.isfile(row[0]) and os.path.isfile(row[1]):
            count = count + 1
            df.append(row)

    print("data processing finished")
    print("data frame size: " + str(count))

    train_generator = utils.train_generator(df, 1)

    inputs, outputs = next(train_generator)

    for i in range(len(utils.labels)):
        plt.imshow(outputs[0, :, :, i], cmap='gray')
        plt.show()

    exit(0)

    # model.load_weights("./enet_v3.h5")
    train(model, epochs=10, train_gen=train_generator, steps_epch=1000, save_path="./enet_v1.h5")
