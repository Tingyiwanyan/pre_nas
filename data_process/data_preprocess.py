import numpy as np
import os
import pickle
import lasagne

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
# Test script

# Change this one to check other file


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_file', help="Input File with images")
    parser.add_argument('-g', '--gen_images', help='If true then generate big (10000 small images on one image) images',
                        action='store_true')
    parser.add_argument('-s', '--sorted_histogram', help='If true then histogram with number of images for '
                                                         'class will be sorted', action='store_true')
    args = parser.parse_args()

    return args.in_file, args.gen_images, args.sorted_histogram


def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
    x = x.reshape((x.shape[0], 64, 64, 3))

    return x, y






