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
import deep_lstm_model_MNIST_dataset
import highway_tranform_lstm_model_MNIST_dataset
import residual_lstm_model_MNIST_dataset

import math

y_bundle = []


batch_size = 300



def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def show_multi_plot(y_bundle, batch_size_):
	indep_test_axis = []
	for i in range(batch_size_):
		indep_test_axis.append(i)
	p = PlotUtil("Accuracy Results on MNIST Dataset", np.array(indep_test_axis), "Epoch iterations", "Accuracy")
	p.show_plot(y_bundle)

def runner_with_results_output(y_bundle):
	test_losses = []
	test_accuracies = []

	for i in range(batch_size):
		test_losses.append(3.5 - 1.6 * sigmoid(i / 10))
		test_accuracies.append(0.5 + 0.4 * sigmoid(i / 10))

	y = YaxisBundle(test_losses, "loss", "b")
	y_bundle.append(y)

	y = YaxisBundle(test_accuracies, "accuracy", "g")
	y_bundle.append(y)
	return y_bundle




if __name__ == '__main__':
	run_with_config = single_layer_lstm_MNIST.run_with_config
	config = single_layer_lstm_MNIST.config

	config.tensor_board_logging_enabled  = False
	config.matplot_lib_enabled = True
	config.matplot_lib_for_single_ybundle = True
	config.matplot_lib_for_accuracy = True
	config.training_epochs = 100
	config.n_stacked_layers = 1
	config.bias_mean = 0.5
	config.learning_rate = 0.0005

	y = run_with_config(config)
	y.y_graph_label = "Vanilla LSTM"
	y.y_graph_colour = "b"
	y_bundle.append(y)

	##### end for model###########

	run_with_config = deep_lstm_model_MNIST_dataset.run_with_config
	config = deep_lstm_model_MNIST_dataset.config

	config.tensor_board_logging_enabled = False
	config.matplot_lib_enabled = True
	config.matplot_lib_for_single_ybundle = True
	config.matplot_lib_for_accuracy = True
	config.training_epochs = 100
	config.n_stacked_layers = 3
	config.bias_mean = 0.5
	config.learning_rate = 0.0005


	y = run_with_config(config)
	y.y_graph_label = "Stacked LSTM"
	y.y_graph_colour = "g"
	y_bundle.append(y)

	##### end for model###########

	run_with_config = highway_tranform_lstm_model_MNIST_dataset.run_with_config
	config = highway_tranform_lstm_model_MNIST_dataset.config

	config.tensor_board_logging_enabled = False
	config.matplot_lib_enabled = True
	config.matplot_lib_for_single_ybundle = True
	config.matplot_lib_for_accuracy = True
	config.training_epochs = 100
	config.n_stacked_layers = 3
	config.bias_mean = 0.5
	config.learning_rate = 0.0005

	y = run_with_config(config)
	y.y_graph_label = "Highway LSTM"
	y.y_graph_colour = "r"
	y_bundle.append(y)

	##### end for model###########

	run_with_config = residual_lstm_model_MNIST_dataset.run_with_config
	config = residual_lstm_model_MNIST_dataset.config

	config.tensor_board_logging_enabled = False
	config.matplot_lib_enabled = True
	config.matplot_lib_for_single_ybundle = True
	config.matplot_lib_for_accuracy = True
	config.training_epochs = 100
	config.n_stacked_layers = 3
	config.bias_mean = 0.5
	config.learning_rate = 0.0005

	y = run_with_config(config)
	y.y_graph_label = "Residual LSTM"
	y.y_graph_colour = "y"
	y_bundle.append(y)

	##### end for model###########


	show_multi_plot(y_bundle, config.training_epochs)
	exit(0)
