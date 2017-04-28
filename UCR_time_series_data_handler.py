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
direc = '/home/red/tf_ver_1/LSTM_with_attention_on_MNIST_/data/UCR_TS_Archive_2015'
summaries_dir = '/home/red/tf_ver_1/LSTM_tsc/LSTM_TSC/log_tb'

def load_data_with_series_size(direc,ratio,dataset):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = direc + '/' + dataset + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
  DATA = np.concatenate((data_train,data_test_val),axis=0)
  N = DATA.shape[0]
  time_series_size = 0

  with open(datadir+'_TRAIN', 'rb') as filep:
    line = filep.readline()
    #print(line)
    line_sp = line.split(",")
    time_series_size = len(line_sp)-1

  ratio = (ratio*N).astype(np.int32)
  ind = np.random.permutation(N)
  X_train = DATA[ind[:ratio[0]],1:]
  X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_test = DATA[ind[ratio[1]:],1:]
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  y_train = DATA[ind[:ratio[0]],0]-1
  y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  y_test = DATA[ind[ratio[1]:],0]-1
  return X_train,X_val,X_test,y_train,y_val,y_test,time_series_size

def load_data(direc,ratio,dataset):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = direc + '/' + dataset + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
  DATA = np.concatenate((data_train, data_test_val),axis=0)
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



def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch



"""Load the data"""
ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
#X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='CBF')
#X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='synthetic_control')



def get_CBF_data(dataset):
  X_train, X_val, X_test, y_train, y_val, y_test, series_size = load_data_with_series_size(direc, ratio, dataset=dataset)
  return X_train, y_train, X_test, y_test

def get_dataset_with_series_size(dataset):
  X_train, X_val, X_test, y_train, y_val, y_test, series_size = load_data_with_series_size(direc, ratio, dataset=dataset)
  return X_train, y_train, X_test, y_test, series_size

if __name__ == '__main__':
    (X_train, y_train, X_test, y_test, series_size) = get_dataset_with_series_size(dataset='synthetic_control')
    print(X_train)