"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  # TF 1.1.0rc1

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
#from tsc_model import Model, sample_batch, load_data

# Set these directories
direc = '/home/red/tf_ver_1/LSTM_tsc/UCR_TS_Archive_2015'
summaries_dir = '/home/red/tf_ver_1/LSTM_tsc/LSTM_TSC/log_tb'

def load_data(direc,ratio,dataset):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = direc + '/' + dataset + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
  DATA = np.concatenate((data_train,data_test_val),axis=0)
  N = DATA.shape[0]

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = DATA[ind[:ratio[0]],1:]
  X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_test = DATA[ind[ratio[1]:],1:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA[ind[:ratio[0]],0]-1
  y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  y_test = DATA[ind[ratio[1]:],0]-1
  return X_train,X_val,X_test,y_train,y_val,y_test

"""Load the data"""
ratio = np.array([0.8, 0.9])  # Ratios where to split the training and validation set
X_train, X_val, X_test, y_train, y_val, y_test = load_data(direc, ratio, dataset='CBF')
# X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='ChlorineConcentration')

def get_CBF_data():
	return (X_train, y_train, X_test, y_test)

N, sl = X_train.shape
num_classes = len(np.unique(y_train))

"""Hyperparamaters"""
batch_size = 30
max_iterations = 3000
dropout = 0.8
config = {'num_layers': 3,  # number of layers of stacked RNN's
          'hidden_size': 120,  # memory cells in a layer
          'max_grad_norm': 5,  # maximum gradient norm during training
          'batch_size': batch_size,
          'learning_rate': .005,
          'sl': sl,
          'num_classes': num_classes}

epochs = np.floor(batch_size * max_iterations / N)
print('Train %.0f samples in approximately %d epochs' % (N, epochs))

if __name__ == '__main__':
    print(get_CBF_data())



