# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from monodepth.monodepth_model import *
from monodepth.average_gradients import *

class monodepth_runner:

    def __init__(self, checkpoint):

        self.params = monodepth_parameters(
            encoder='vgg',
            height=256,
            width=512,
            batch_size=2,
            num_threads=1,
            num_epochs=1,
            do_stereo=False,
            wrap_mode="border",
            use_deconv=False,
            alpha_image_loss=0,
            disp_gradient_loss_weight=0,
            lr_loss_weight=0,
            full_summary=False)

        self.input_height = 256
        self.input_width = 512
        self.checkpoint_path = checkpoint

        # Initializing monodepth model
        self.left = tf.placeholder(tf.float32, [2, self.input_height, self.input_width, 3])
        self.model = MonodepthModel(self.params, "test", self.left, None)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # RESTORE
        restore_path = self.checkpoint_path.split(".")[0]
        train_saver.restore(self.sess, restore_path)

    @staticmethod
    def post_process_disparity(disp):
        _, h, w = disp.shape
        l_disp = disp[0, :, :]
        r_disp = np.fliplr(disp[1, :, :])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)

        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def run_depth(self, image_path, out_height, out_width):

        """Test function."""
        input_image = scipy.misc.imread(image_path, mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [self.input_height, self.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = self.sess.run(self.model.disp_left_est[0], feed_dict={self.left: input_images})
        disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [out_height, out_width])

        return disp_to_img
