#!/usr/bin/python

import argparse
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import utils
import cv2

from models.icnet import ICNet
import configs

#### Test ####

# Workaround to forbid tensorflow from crashing the gpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.27
set_session(tf.Session(config=config))

# Model
net = ICNet(width=configs.img_width, height=configs.img_height, n_classes=34, weight_path="output/icnet_large_full_040_0.781.h5", training=False)
print(net.model.summary())

# Testing
x = cv2.resize(cv2.imread("./testing_imgs/10.png", 1), (configs.img_width, configs.img_height))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

mid = cv2.resize(x, (configs.img_width / 2, configs.img_height / 2))
x = np.array([x])
y = net.model.predict(x)[0]

start_time = time.time()
for i in range(50):
    y = net.model.predict(x)[0]
duration = time.time() - start_time
print('Generated segmentations in %s seconds -- %s FPS' % (duration / 50, 1.0/(duration/50)))

y = cv2.resize(y, (configs.img_width / 2, configs.img_height / 2))
image = utils.convert_class_to_rgb(y, threshold=0.50)
viz = image # cv2.addWeighted(mid, 0.8, image, 0.8, 0)
plt.figure(1)
plt.imshow(viz)
plt.show()

cv2.imwrite('seg_result_overlay.png', cv2.resize(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR), (1024, 512)))
