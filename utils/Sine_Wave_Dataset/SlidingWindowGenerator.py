#
# Author : Praveen
import numpy as np

class SlidingWindowGenerator(object):
	"""
	Class for generating Time-series Dataset using sliding window algorithm.

	>>> import numpy as np
	>>> X = np.array([0, 1, 7, 3, 4,])
	>>> print SlidingWindowGenerator(X,window_size=3).runSlidingWindow()
	[array([0, 1, 7]), array([ 1, 7, 3], array([7, 3, 4]))]
	
	"""	
	def __init__(self, data, window_size = 10):
		self.window_size = window_size
		self.data = data ##TODO assert data is np.array

	def runSlidingWindow(self, max_dataset_size = 0 ):
		output = []
		range_len = len(X) - self.window_size
		if(max_dataset_size != 0 and range_len  >= max_dataset_size -1):
			range_len = max_dataset_size - 1
		for i in range(range_len+1):
			output.append(X[i: i+self.window_size])
		return output

if __name__ == '__main__':
	X = np.array([0, 1, 7, 3, 4,])
	print (SlidingWindowGenerator(X, window_size=3).runSlidingWindow(max_dataset_size = 1))
