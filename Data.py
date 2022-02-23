#!/usr/local/bin/python3

import tensorflow as tf
from matplotlib import pyplot
import numpy as np

class Data(object):
    """
    Stores data information for use test and training data
    """
    def __init__(self):
        # import dataset from keras
        data = self.import_digit_dataset()
        self.x_train = data[0][0]
        self.y_train = data[0][1]
        self.x_test = data[1][0]
        self.y_test = data[1][1]

        # Normalize dataset to a 0-1 scale
        self.x_train = self.normalize(self.x_train)
        self.x_test = self.normalize(self.x_test)

        # Flatten dataset into list of 1D arrays
        self.x_train_flattened = self.flatten(self.x_train)
        self.x_test_flattened = self.flatten(self.x_test)

    # ------- static methods ---------
    @staticmethod
    def import_digit_dataset():
        """
        imports keras dataset for handwritten digits
        """
        return tf.keras.datasets.mnist.load_data(path="mnist.npz")

    @staticmethod
    def normalize(input):
        """
        Normalizes dataset to float from 0-1
        """
        return input/255.0

    @staticmethod
    def flatten(input):
        """
        Flattens 2D image array into 1D array
        """
        flattened_arr = []
        for data in input:
            flattened_arr.append(data.flatten())
        return np.array(flattened_arr)

    @staticmethod
    def encode_results(input):
        """
        Encodes results into 1-hot list
        """
        return np.eye(10)[input]

    @staticmethod
    def split(lst, n):
        """Yield successive n-sized chuncks from lst"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # ------- class methods ----------
    def print_data(self, data_type, index):
        """
        prints the data in console as acii characters for quick debug
        """
        str = ''
        if data_type == 'train':
            data = self.x_train
            ans = self.y_train
        elif data_type == 'test':
            data = self.x_test
            ans = self.y_test
        else:
            return
        print(ans[index])
        for x in range(len(data[index])):
            for y in range(len(data[index][x])):
                if data[index][x][y] > 0.66:
                    str += ' #'
                elif data[index][x][y] > 0.33:
                    str += ' /'
                elif data[index][x][y] > 0:
                    str += ' .'
                else:
                    str += '  '
            str += '\n'
        print(str)

