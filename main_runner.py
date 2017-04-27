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
tf.app.flags.DEFINE_string('data_dir', '',
                           'where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('batch_size', 10, 'use batch normalization. otherwise use biases')
FLAGS = tf.flags.FLAGS





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


class HARConfig(Config):
	def __init__(self):
		super(HARConfig, self).__init__()
		self.train_count = 7352  # 7352 training series
		self.test_data_count = 2947  # 2947 testing series
		self.n_steps = 128 # 128 time_steps per series
		self.n_classes = 6  # Final output classes

		# Training
		self.learning_rate = 0.001
		self.lambda_loss_amount = 0.005
		self.training_epochs = 250
		self.batch_size = 100
		self.clip_gradients = 15.0
		self.gradient_noise_scale = None
		# Dropout is added on inputs and after each stacked layers (but not
		# between residual layers).
		self.keep_prob_for_dropout = 0.85  # **(1/3.0)

		# Linear+relu structure
		self.bias_mean = 0.3
		# I would recommend between 0.1 and 1.0 or to change and use a xavier
		# initializer
		self.weights_stddev = 0.2

		########
		# NOTE: I think that if any of the below parameters are changed,
		# the best is to readjust every parameters in the "Training" section
		# above to properly compare the architectures only once optimised.
		########

		# LSTM structure
		# Features count is of 9: three 3D sensors features over time
		self.n_inputs = 0
		self.n_hidden = 12  # 28  # nb of neurons inside the neural network
		# Use bidir in every LSTM cell, or not:
		self.use_bidirectionnal_cells = False

		# High-level deep architecture
		self.also_add_dropout_between_stacked_cells = False  # True
		self.n_layers_in_highway = 1
		self.n_stacked_layers = 3



run_with_config = deep_lstm_model.run_with_config



for learning_rate in [0.0001, 0.002]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for decay in [0.9]: #[0.005, 0.01]:
		config = HARConfig()
		#config = Config()
		config.learning_rate = learning_rate
		config.decay = decay
		run_with_config(config) #, trX, trY, teX, teY)










if __name__ == '__main__':
    pass
