import matplotlib
import numpy as np
from matplotlib import pyplot
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))



#print(sigmoid(0))


# To keep track of training's performance
test_losses = []
test_accuracies = []

indep_test_axis = []

width = 6#12
height = 6#12
pyplot.figure(figsize=(width, height))
batch_size = 300
training_iters = 8000 * 300  # Loop 300 times on the dataset
display_iter = 30000


for i in range(batch_size):
    indep_test_axis.append(i)
    test_losses.append(3.5 -  1.6 * sigmoid( i/10))
    test_accuracies.append(0.5 + 0.4 * sigmoid(i/10))
#print(test_losses)
#print (test_accuracies)

class YaxisBundle:
	def __init__(self, y_index_array_input , y_graph_colour):
		self.y_index_array_input  = y_index_array_input
		self.y_graph_colour = y_graph_colour
		self.y_label =  y_label


class PlotUtil:
	def __init__(self, plot_title, x_index_array_input , x_label, width = 6, height = 6):
		self.x_index_array_input  = x_index_array_input
		self.plot_title = plot_title
		self.x_label =  x_label
		self.width = width
		self.height = height
	def show_plot(self, y_target):
		pyplot.figure(figsize=(width, height))
		pyplot.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
		pyplot.title("Training session's progress over iterations")
		pyplot.legend(loc='upper right', shadow=True)
		pyplot.ylabel('Training Progress (Loss or Accuracy values)')
		pyplot.xlabel('Training iteration')

		pyplot.show()






if __name__ == '__main__':
    """
     comments
    """
    pass
