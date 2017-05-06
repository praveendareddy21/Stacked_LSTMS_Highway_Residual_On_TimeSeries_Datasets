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
		self.batch_norm_enabled = False
		self.n_stacked_layers = 1
		self.training_epochs = 300
		self.batch_size = 1500
		self.tensor_board_logging_enabled = False
		self.model_name = "base_config"
		self.log_folder_suffix = "base_config"


	def print_config(self):
		print("#####")
		print("learning_rate" +" : "+ str(self.learning_rate))
		print("decay" +" : "+ str(self.decay))
		print("batch_norm" + " : " + str(self.batch_norm_enabled))
		print("n_stacked_layers" + " : " + str(self.n_stacked_layers))
		print("training_epochs" + " : " + str(self.training_epochs))
		print("batch_size" + " : " + str(self.batch_size))

	def attach_log_suffix(self):
		log_suffix = ""
		log_suffix = log_suffix + "model:" + str(self.model_name)
		log_suffix = log_suffix + "/" + "learn:" + str(self.learning_rate)
		log_suffix = log_suffix + "/" +"stacked_layer:" + str(self.n_stacked_layers)
		log_suffix = log_suffix + "/" + "epochs:" + str(self.training_epochs)

		return log_suffix

		#self.random = Random(FLAGS.python_seed)





