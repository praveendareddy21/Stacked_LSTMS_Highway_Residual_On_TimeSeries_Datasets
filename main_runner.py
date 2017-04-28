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




FLAGS = tf.app.flags.FLAGS





class Config(object):
	def __init__(self):
		global FLAGS
		self.FLAGS = FLAGS
		self.tf_data_type = {}
		self.tf_data_type["double"] = tf.float64
		self.tf_data_type["float"] = tf.float32
		self.np_data_type = {}
		self.np_data_type["double"] = np.float64
		self.np_data_type["float"] = np.float32
		self.learning_rate = 0.001
		self.decay = 0.9

	def print_config(self):
		print("learning_rate" +" : "+ str(self.learning_rate))
		print("decay" +" : "+ str(self.decay))

    #self.random = Random(FLAGS.python_seed)
















if __name__ == '__main__':

	run_with_config = deep_lstm_model.run_with_config
	config = deep_lstm_model.config


	for learning_rate in [0.0001, 0.002]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
		for decay in [0.9]: #[0.005, 0.01]:
			config.learning_rate = learning_rate
			config.decay = decay
			run_with_config(config) #, trX, trY, teX, teY)
