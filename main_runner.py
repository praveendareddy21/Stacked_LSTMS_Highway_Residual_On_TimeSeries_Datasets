import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops


import lstm_mnist
import bi_dir_residual_lstm_HAR


import deep_lstm_model
import single_layer_lstm
import deep_lstm_model_with_BN
import deep_lstm_model_on_ucr_dataset
import highway_lstm_model
import residual_lstm_model




if __name__ == '__main__':

	run_with_config = residual_lstm_model.run_with_config
	config = residual_lstm_model.config

	#run_with_config = deep_lstm_model.run_with_config
	#config = deep_lstm_model.config

	#run_with_config = deep_lstm_model_on_ucr_dataset.run_with_config
	#config = deep_lstm_model_on_ucr_dataset.config

	#run_with_config = deep_lstm_model_with_BN.run_with_config
	#config = deep_lstm_model_with_BN.config


	for learning_rate in [0.005, 0.0025, 0.003]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
		for decay in [0.9]: #[0.005, 0.01]:
			for bn_enabled in [True, False]:
				for n_stacked in [1]: # [2,3, 6]:
					config.tensor_board_logging_enabled = False #should be always False, log summary folder gets impacted by mulitple runs
					config.n_stacked_layers = n_stacked
					config.batch_norm_enabled = bn_enabled
					config.learning_rate = learning_rate
					config.decay = decay
					run_with_config(config) #, trX, trY, teX, teY)
