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



if __name__ == '__main__':

	#run_with_config = single_layer_lstm.run_with_config
	#config = single_layer_lstm.config

	run_with_config = deep_lstm_model.run_with_config
	config = deep_lstm_model.config

	run_with_config = deep_lstm_model_with_BN.run_with_config
	config = deep_lstm_model_with_BN.config



	for learning_rate in [0.0001, 0.002]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
		for decay in [0.9]: #[0.005, 0.01]:
			config.learning_rate = learning_rate
			config.decay = decay
			run_with_config(config) #, trX, trY, teX, teY)
