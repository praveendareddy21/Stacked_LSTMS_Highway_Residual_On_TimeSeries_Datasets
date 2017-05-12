import tensorflow as tf
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
import matplotlib
from matplotlib import pyplot
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))





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

		self.train_count = 0
		self.test_data_count = 0
		self.n_steps = 0


		# LSTM structure
		self.n_inputs = 0  # == 9 Features count is of 9: three 3D sensors features over time
		self.n_hidden = 0  # nb of neurons inside the neural network
		self.n_classes = 0 # Final output classes

		self.matplot_lib_enabled = False
		self.matplot_lib_for_accuracy = True


	def print_config(self):
		print("#####")
		print("learning_rate" +" : "+ str(self.learning_rate))
		print("decay" +" : "+ str(self.decay))
		print("batch_norm" + " : " + str(self.batch_norm_enabled))
		print("n_stacked_layers" + " : " + str(self.n_stacked_layers))
		print("training_epochs" + " : " + str(self.training_epochs))
		print("batch_size" + " : " + str(self.batch_size))
		print("model_name" + " : " + str(self.model_name))

		print("train dataset size" + " : " + str(self.train_count))
		print("test dataset size" + " : " + str(self.test_data_count))
		print("time-series size" + " : " + str(self.n_steps))

		print("lstm neuron count" + " : " + str(self.n_hidden))
		print("output class count" + " : " + str(self.n_classes))

	def attach_log_suffix(self):
		log_suffix = ""
		log_suffix = log_suffix + "model:" + str(self.model_name)
		log_suffix = log_suffix + "/" + "learn:" + str(self.learning_rate)
		log_suffix = log_suffix + "/" +"stacked_layer:" + str(self.n_stacked_layers)
		log_suffix = log_suffix + "/" + "epochs:" + str(self.training_epochs)

		return log_suffix

		#self.random = Random(FLAGS.python_seed)
		
		
		
		

class YaxisBundle:
	def __init__(self, y_index_array_input, y_graph_label, y_graph_colour):
		self.y_index_array_input  = y_index_array_input
		self.y_graph_colour = y_graph_colour
		self.y_graph_label = y_graph_label



class PlotUtil:
	def __init__(self, plot_title, x_index_array_input , x_label, y_label, width = 6, height = 6):
		self.x_index_array_input  = x_index_array_input
		self.plot_title = plot_title
		self.x_label =  x_label
		self.y_label =  y_label
		self.width = width
		self.height = height
	def show_plot(self, YaxisBundle_array, is_x_index_variable = True):
		pyplot.figure(figsize=(self.width, self.height))
		assert isinstance(self.x_index_array_input, np.ndarray), "X-axis index must be a numpy ndarray"
		x_axis_length = len(self.x_index_array_input)


		for y in YaxisBundle_array:
			assert isinstance(y.y_index_array_input, np.ndarray), "Y-axis array must be a numpy ndarray"
			if is_x_index_variable: #TODO refine this logic
				self.x_index_array_input = None
				_ = []
				for i in range(len(y.y_index_array_input)):
					_.append(i)
				self.x_index_array_input = np.array(_)

			assert (len(y.y_index_array_input) == len(self.x_index_array_input)), "Both axes indexes must be of same length"

			pyplot.plot(self.x_index_array_input, y.y_index_array_input, y.y_graph_colour, label=y.y_graph_label)

		pyplot.title(self.plot_title)
		pyplot.legend(loc='upper right', shadow=True)
		pyplot.ylabel(self.y_label)
		pyplot.xlabel(self.x_label)
		pyplot.show()

		
if __name__ == '__main__':

	test_losses = []
	test_accuracies = []
	indep_test_axis = []
	batch_size = 300

	for i in range(batch_size):
	    indep_test_axis.append(i)
	    test_losses.append(3.5 -  1.6 * sigmoid( i/10))
	    test_accuracies.append(0.5 + 0.4 * sigmoid(i/10))

	indep_test_axis = np.array(indep_test_axis)
	test_losses = np.array(test_losses)
	test_accuracies = np.array(test_accuracies)


	p = PlotUtil("title", indep_test_axis, "x_label", "y_label")
	y_bundle =[]
	
	y = YaxisBundle(test_losses,"loss", "b")
	y_bundle.append(y)

	y = YaxisBundle(test_accuracies,"accuracy", "g")
	y_bundle.append(y)
	
	p.show_plot(y_bundle)



