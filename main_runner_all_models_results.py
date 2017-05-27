import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
from base_config import PlotUtil, YaxisBundle


import lstm_mnist



import deep_lstm_model
import single_layer_lstm
import highway_lstm_model
import residual_lstm_model
import  deep_lstm_model_UCR_dataset
import highway_lstm_model_UCR_dataset
import highway_lstm_model_plotlib

import single_layer_lstm_MNIST
import highway_tranform_lstm_model
import residual_lstm_model_MNIST_dataset

import math

y_bundle = []


learning_rate  = 300
layers = 6
_matplot_lib_enabled = False

_bias_mean = 0.5
_learning_rate = 0.005


if __name__ == '__main__':
	run_with_config = single_layer_lstm.run_with_config
	config = single_layer_lstm.config

	config.tensor_board_logging_enabled  = False
	config.matplot_lib_enabled = _matplot_lib_enabled

	config.matplot_lib_for_single_ybundle = True
	config.matplot_lib_for_accuracy = True
	config.n_stacked_layers = 1


	config.bias_mean = _bias_mean
	config.learning_rate = _learning_rate

	y = run_with_config(config)

	##### end for model###########

	run_with_config = deep_lstm_model.run_with_config
	config = deep_lstm_model.config

	config.matplot_lib_enabled = _matplot_lib_enabled


	config.tensor_board_logging_enabled = False
	config.n_stacked_layers = layers

	config.bias_mean = _bias_mean
	config.learning_rate = _learning_rate


	y = run_with_config(config)

	##### end for model###########

	run_with_config = highway_tranform_lstm_model.run_with_config
	config = highway_tranform_lstm_model.config


	config.matplot_lib_enabled = _matplot_lib_enabled

	config.tensor_board_logging_enabled = False
	config.n_stacked_layers = layers
	config.matplot_lib_enabled = _matplot_lib_enabled

	config.bias_mean = _bias_mean
	config.learning_rate = _learning_rate

	y = run_with_config(config)

	##### end for model###########

	run_with_config = residual_lstm_model.run_with_config
	config = residual_lstm_model.config

	config.tensor_board_logging_enabled = False
	config.n_stacked_layers = layers
	config.matplot_lib_enabled = _matplot_lib_enabled

	config.bias_mean = _bias_mean
	config.learning_rate = _learning_rate


	y = run_with_config(config)

	##### end for model###########


	exit(0)


