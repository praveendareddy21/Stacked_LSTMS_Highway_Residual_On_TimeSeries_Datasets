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
import deep_lstm_model_on_ucr_dataset
import highway_lstm_model
import residual_lstm_model
import  deep_lstm_model_UCR_dataset
import highway_lstm_model_UCR_dataset
import math

y_bundle = []
indep_test_axis_0 = []

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def show_multi_plot():

	test_losses = []
	test_accuracies = []
	indep_test_axis = []
	batch_size = 300

	for i in range(batch_size):
		indep_test_axis.append(i)
		test_losses.append(3.5 - 1.6 * sigmoid(i / 10))
		test_accuracies.append(0.5 + 0.4 * sigmoid(i / 10))

	p = PlotUtil("title", indep_test_axis, "x_label", "y_label")
	y_bundle = []

	y = YaxisBundle(test_losses, "loss", "b")
	y_bundle.append(y)

	y = YaxisBundle(test_accuracies, "accuracy", "g")
	y_bundle.append(y)

	p.show_plot(y_bundle)


if __name__ == '__main__':
	show_multi_plot()
	exit(0)

	#run_with_config = deep_lstm_model.run_with_config
	#config = deep_lstm_model.config

	"""
	for training_epochs_ in [10, 20]:
		config.tensor_board_logging_enabled = False  # should be always False, log summary folder gets impacted by mulitple runs
		config.training_epochs = training_epochs_
		run_with_config(config)
	exit(0)
	"""
	#run_with_config = deep_lstm_model.run_with_config
	#config = deep_lstm_model.config

	#run_with_config = deep_lstm_model_on_ucr_dataset.run_with_configp
	#config = deep_lstm_model_on_ucr_dataset.config

	run_with_config = highway_lstm_model_UCR_dataset.run_with_config
	config = highway_lstm_model_UCR_dataset.config


	for learning_rate in [0.005, 0.0025, 0.003, 0.0005]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
		for decay in [0.9]: #[0.005, 0.01]:
			for bn_enabled in [True, False]:
				for n_stacked in [2,3, 6]:
					for epoch_count in [200, 300, 450]:
						config.training_epochs = epoch_count
						config.tensor_board_logging_enabled = False #should be always False, log summary folder gets impacted by mulitple runs
						config.n_stacked_layers = n_stacked
						config.batch_norm_enabled = bn_enabled
						config.learning_rate = learning_rate
						config.decay = decay
						run_with_config(config) #, trX, trY, teX, teY)
