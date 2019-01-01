# 
# utilities for semantic segmentation
# autonomous golf cart project
# (c) Yongyang Nie, Michael Meng
# ==============================================================================
#

import cv2
import configs as configs
import numpy as np
import os
from collections import namedtuple

Label = namedtuple('Label', [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ])


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,       (  0,  0,  0) ),
    Label(  'ground'               ,  6 ,       ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,       (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,       (244, 35,232) ),
    Label(  'building'             , 11 ,       ( 70, 70, 70) ),
    Label(  'pole'                 , 17 ,       (153,153,153) ),
    Label(  'vegetation'           , 21 ,       (107,142, 35) ),
    Label(  'terrain'              , 22 ,       (152,251,152) ),
    Label(  'sky'                  , 23 ,       ( 70,130,180) ),
    Label(  'person'               , 24 ,       (220, 20, 60) ),
    Label(  'car'                  , 26 ,       (  0,  0,142) ),
    Label(  'motorcycle'           , 32 ,       (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       (119, 11, 32) ),
]

def bc_img(img, s = 1.0, m = 0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def prepare_dataset(path):

    inputs = os.listdir(path)
    imgs = os.listdir(path)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:-11] + "_road" + imgs[i][-11:]

    return inputs, imgs


def load_image(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (configs.img_width, configs.img_height))

    return img


def convert_rgb_to_class(image):

    outputs = []

    for i in range(len(labels)):

        label = labels[i]
        color = np.array(label[2], dtype=np.uint8)
        # objects found in the frame.
        mask = cv2.inRange(image, color, color)

        # divide each pixel by 255
        mask = np.true_divide(mask, 255)

        if len(outputs) == 0:
            outputs = mask
        else:
            outputs = np.dstack((outputs, mask))

    return outputs


def convert_class_to_rgb(image_labels, threshold=0.25):

    # convert any pixel > threshold to 1
    # convert any pixel < threshold to 0
    # then use bitwise_and

    output = np.zeros((configs.img_height, configs.img_width, 3), dtype=np.uint8)

    for i in range(len(labels)):

        split = image_labels[:, :, i]
        split[split > threshold] = 1
        split[split < threshold] = 0
        split[:] *= 255
        split = split.astype(np.uint8)
        color = labels[i][7]

        bg = np.zeros((configs.img_height, configs.img_width, 3), dtype=np.uint8)
        bg[:, :, 0].fill(color[0])
        bg[:, :, 1].fill(color[1])
        bg[:, :, 2].fill(color[2])

        res = cv2.bitwise_and(bg, bg, mask=split)

        output = cv2.addWeighted(output, 1.0, res, 1.0, 0)


    return output


def validation_generator(labels, batch_size):

    batch_images = np.zeros((batch_size, configs.img_height, configs.img_width, 3))
    batch_masks = np.zeros((batch_size, configs.img_height, configs.img_width, 3))

    while 1:

        for index in np.random.permutation(len(labels)):

            label = labels[index]
            image = load_image(configs.data_path + "leftImg8bit/val/" + label[1])
            gt_image = load_image(configs.data_path + "gtFine/val/" + label[2])

            batch_images[index] = image
            batch_masks[index] = gt_image

        yield batch_images, batch_masks


def train_generator(df, batch_size):

    """
    An important method that returns the generator used
    for training the segmentation network

    :param df: data frame, the loaded csv data
    :param batch_size: training batch size
    :return: training generator
    """

    batch_images = np.zeros((batch_size, configs.img_height, configs.img_width, 3))
    batch_masks = np.zeros((batch_size, configs.img_height, configs.img_width, len(labels)))

    while 1:
        i = 0
        for index in np.random.permutation(len(df)):

            label = df[index]

            image = np.array(load_image(label[0]), dtype=np.float32) / 255
            gt_image = load_image(label[1])
            batch_images[i] = image
            batch_masks[i] = convert_rgb_to_class(gt_image)

            i += 1
            if i == batch_size:
                break

        yield batch_images, batch_masks


"""
The main method is used 
for testing the helper methods
"""

if __name__ == "__main__":

    img = load_image("./testing_imgs/test.png")
    print(img.shape)
    array = convert_rgb_to_class(img)
    image = convert_class_to_rgb(array)
