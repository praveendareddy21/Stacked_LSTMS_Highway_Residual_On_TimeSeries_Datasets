import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops


import lstm_mnist



import deep_lstm_model
import single_layer_lstm
import highway_lstm_model
import residual_lstm_model
import  deep_lstm_model_UCR_dataset
import highway_lstm_model_UCR_dataset




if __name__ == '__main__':
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

	run_with_config = single_layer_lstm.run_with_config
	config = single_layer_lstm.config


	for learning_rate in [0.005, 0.0025, 0.003, 0.0005]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
		for decay in [0.9]: #[0.005, 0.01]:
			for bn_enabled in [True, False]:
				for n_stacked in [1]: #2 3 6
					for epoch_count in [200, 300, 450]:
						config.training_epochs = epoch_count
						config.tensor_board_logging_enabled = False #should be always False, log summary folder gets impacted by mulitple runs
						config.n_stacked_layers = n_stacked
						config.batch_norm_enabled = bn_enabled
						config.learning_rate = learning_rate
						config.decay = decay
						run_with_config(config) #, trX, trY, teX, teY)
