import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '',
                           'where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('batch_size', 10, 'use batch normalization. otherwise use biases')
FLAGS = tf.flags.FLAGS


class Utility:
  #holds FLAGS and other variables that are used in different files
  def __init__(self):
    global FLAGS
    self.FLAGS = FLAGS
    self.tf_data_type = {}
    self.tf_data_type["double"] = tf.float64
    self.tf_data_type["float"] = tf.float32
    self.np_data_type = {}
    self.np_data_type["double"] = np.float64
    self.np_data_type["float"] = np.float32

    #self.random = Random(FLAGS.python_seed)

utility = Utility()

print("batch_size : "+ str(FLAGS.batch_size))
# model training
mnist = input_data.read_data_sets("data", one_hot=True)
sess = tf.InteractiveSession()

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [utility.FLAGS.batch_size, 10])

"""
label_predict = model(image)

cross_entropy = -tf.reduce_sum(label * tf.log(label_predict))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(label_predict, 1), tf.arg_max(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
"""