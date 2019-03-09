#
# config.py: configuration for semantic segmentation
# (c) Neil Nie, 2017
# All Rights Reserved.
#

img_height = 512
img_width = 1024
learning_rate = 1e-4

val_depth_color_path = '/home/neil/Workspace/semantic-segmentation/new_labels.csv'
label_depth_color_path = '/home/neil/Workspace/semantic-segmentation/new_val_labels.csv'
data_path = '/hdd/ssd_2/dataset/segmentation/train_labels.csv'
labelid_path = '/hdd/ssd_2/dataset/segmentation/labels.csv'
val_label_path = '/hdd/ssd_2/dataset/segmentation/val_labels.csv'
infer_model_path = './weights/enet-c-v1-3.h5'
test_dataset = "/Volumes/Personal_Drive/Datasets/CityScapes/"
