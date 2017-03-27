import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
from lstm_mnist import run_with_config

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '',
                           'where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('batch_size', 10, 'use batch normalization. otherwise use biases')
FLAGS = tf.flags.FLAGS





class Config:
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





for learning_rate in [0.0001, 0.002]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for decay in [0.9]: #[0.005, 0.01]:
    	config = Config()
    	config.learning_rate = learning_rate
    	config.decay = decay
       	run_with_config(config) #, trX, trY, teX, teY)










if __name__ == '__main__':
    pass
