#!/usr/bin/python

import argparse
import time
import json

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from keras import optimizers
import utils
import cv2

from models.icnet import ICNet
from utils import apply_color_map


#### Test ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Model
optim = optimizers.SGD(lr=0.01, momentum=0.9)
net = ICNet(width=512, height=512, n_classes=13, weight_path='./icnet2-v6.h5', training=False)
net.model.compile(optim, 'categorical_crossentropy', metrics=['categorical_accuracy'])

print(net.model.summary())
# Testing
x = cv2.resize(cv2.imread("./testing_imgs/test_6.jpg", 1), (512, 512))
x = np.array([x])
start_time = time.time()
y = net.model.predict(x)[0]
y = cv2.resize(y, (512, 512))
image = utils.convert_class_to_rgb(y, threshold=0.25)
plt.imshow(image)
plt.show()
# cv2.imwrite('output/output_sample.png', cv2.resize(y, (512, 512)))
duration = time.time() - start_time

print('Generated segmentations in %s seconds -- %s FPS' % (duration, 1.0/duration))

# Save output image
with open('datasets/mapillary/config.json') as config_file:
    config = json.load(config_file)
labels = config['labels']

output = apply_color_map(np.argmax(y[0], axis=-1), labels)
cv2.imwrite('output/output_sample.png', cv2.resize(output, (512, 512)))
###############
