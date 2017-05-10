import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
import os
from UCR_time_series_data_handler import get_dataset_with_series_size, direc
import lstm_mnist



import deep_lstm_model
import single_layer_lstm
import deep_lstm_model_on_ucr_dataset
import highway_lstm_model
import residual_lstm_model
import  deep_lstm_model_UCR_dataset
import highway_lstm_model_UCR_dataset

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_dataset_list():
    return get_immediate_subdirectories(direc)

def run_with_dataset(dataset):
    pass

def modify_config_wtih_current_dataset(config):
    pass

if __name__ == '__main__':
    print (get_dataset_list())