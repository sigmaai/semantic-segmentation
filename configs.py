#
# config.py: configuration for semantic segmentation
# (c) Neil Nie, 2017
# All Rights Reserved.
#

batch_size = 8
img_height = 1024
img_width = 2048
learning_rate = 1e-4

test_results = False
visualize_gen = False
epochs = 2

data_path = '/hdd/ssd_2/dataset/segmentation/train_labels.csv'
labelid_path = '/hdd/ssd_2/dataset/segmentation/labels.csv'
val_label_path = '/hdd/ssd_2/dataset/segmentation/val_labels.csv'
infer_model_path = './weights/enet-c-v1-3.h5'
test_dataset = "/Volumes/Personal_Drive/Datasets/CityScapes/"
