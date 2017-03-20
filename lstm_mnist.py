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


input_vec_size = lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)


def run_with_config(Config, trX, trY, teX, teY):
	Config.print_config()
	ops.reset_default_graph()
	with tf.device("/cpu:0"):  # Remove this line to use GPU. If you have a too small GPU, it crashes.
		X = tf.placeholder("float", [None, 28, 28])
		Y = tf.placeholder("float", [None, 10])

		# get lstm_size and output 10 labels
		W = init_weights([lstm_size, 10])
		B = init_weights([10])

		py_x, state_size = model(X, W, B, lstm_size)


		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
		train_op = tf.train.RMSPropOptimizer(Config.learning_rate, Config.decay).minimize(cost)
		predict_op = tf.argmax(py_x, 1)

	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True

	# Launch the graph in a session
	with tf.Session(config=session_conf) as sess:
	    # you need to initialize all variables
	    tf.global_variables_initializer().run()

	    for i in range(5): # was 100 iter
	        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
	            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

	        test_indices = np.arange(len(teX))  # Get A Test Batch
	        np.random.shuffle(test_indices)
	        test_indices = test_indices[0:test_size]

	        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
	                         sess.run(predict_op, feed_dict={X: teX[test_indices]})))
