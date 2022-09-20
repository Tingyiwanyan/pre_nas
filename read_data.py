from data_process import data_preprocess as dp
from matplotlib import pyplot as plt
from PIL import Image
import cv2

file_path_train_32 = '/home/tingyi/Downloads/img_net_subset/Imagenet32_train/train_data_batch_1'

file_path_val_32 = '/home/tingyi/Downloads/img_net_subset/Imagenet32_val/val_data'

if __name__ == '__main__':

    x_train, y_train = dp.load_data(file_path_train_32)
    x_val, y_val = dp.load_data(file_path_val_32)