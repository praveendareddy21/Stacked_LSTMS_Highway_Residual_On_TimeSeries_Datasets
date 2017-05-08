import matplotlib
import numpy as np
from matplotlib import pyplot
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))




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
	def show_plot(self, YaxisBundle_array):
		pyplot.figure(figsize=(self.width, self.height))

		for y in YaxisBundle_array:
			pyplot.plot(self.x_index_array_input, np.array(y.y_index_array_input), y.y_graph_colour, label=y.y_graph_label)

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


	p = PlotUtil("title", indep_test_axis, "x_label", "y_label")
	y_bundle =[]
	
	y = YaxisBundle(test_losses,"loss", "b")
	y_bundle.append(y)

	y = YaxisBundle(test_accuracies,"accuracy", "g")
	y_bundle.append(y)
	
	p.show_plot(y_bundle)
