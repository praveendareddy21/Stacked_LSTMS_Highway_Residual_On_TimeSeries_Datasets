import tensorflow as tf
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops






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





