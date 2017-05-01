#
# Author : Praveen
import numpy as np

class SlidingWindowGenerator(object):
"""
Class for generating Time-series Dataset using sliding window algorithm.

>>> import numpy as np
>>> X = np.array([0, 1, 7, 3, 4,])
>>> print SlidingWindowGenerator(window_size=3, X).runSlidingWindow()
[array([0, 1, 7]), array([ 1, 7, 3], array([7, 3, 4]))]

"""	def __init__(self, window_size = 10, data):
		self.window_size = window_size
		self.data = data ##TODO assert data is np.array

	def runSlidingWindow():
		output = []
		return output
