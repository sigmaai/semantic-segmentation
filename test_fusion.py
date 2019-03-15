#!/usr/bin/python

import argparse
import time
import json

import numpy as np
import matplotlib.pyplot as plt
from monodepth.monodepth_runner import monodepth_runner
from keras.backend.tensorflow_backend import set_session
import utils
import cv2

from models.icnet_fusion import ICNet
import configs

#### Test ####

# define global variables
checkpoint_path = '/home/neil/Workspace/semantic-segmentation/monodepth/models/cityscape/model_cityscapes.data-00000-of-00001'
model_path = 'icnet_early_fusion_train_2_030_0.866.h5'
test_img_path = "./testing_imgs/10.png"

# ==== create monodepth runner ====
depth_runner = monodepth_runner(checkpoint_path)

# ====== Model ======
net = ICNet(width=configs.img_width, height=configs.img_height, n_classes=34, weight_path="output/" + model_path)
print(net.model.summary())

# ======== Testing ========
x = cv2.resize(cv2.imread(test_img_path, 1), (configs.img_width, configs.img_height))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x_depth = depth_runner.run_depth(image_path=test_img_path, out_height=configs.img_height, out_width=configs.img_width)
x_depth = np.dstack((x_depth, x_depth, x_depth))

plt.imshow(x_depth)
plt.show()

mid = cv2.resize(x, (configs.img_width / 2, configs.img_height / 2))
x = np.array([np.concatenate((x, x_depth), axis=2)])

y = net.model.predict(x)[0]

# ===== running... =====
start_time = time.time()
for i in range(10):
    y = net.model.predict(x)[0]

duration = time.time() - start_time
print('Generated segmentations in %s seconds -- %s FPS' % (duration / 10, 1.0/(duration/10)))

# ===== visualization =====
y = cv2.resize(y, (configs.img_width / 2, configs.img_height / 2))
image = utils.convert_class_to_rgb(y, threshold=0.50)
viz = cv2.addWeighted(mid, 0.8, image, 0.8, 0)
plt.figure(1)
plt.imshow(viz)
plt.show()

cv2.imwrite('seg_result_overlay.png', cv2.resize(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR), (1024, 512)))
