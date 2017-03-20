import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '',
                           'where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('batch_size', 10, 'use batch normalization. otherwise use biases')
FLAGS = tf.flags.FLAGS

#holds FLAGS and other variables that are used in different files
class Config:
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

	def print_config(self):
		print("learning_rate" +" : "+ str(self.learning_rate))
		print("decay" +" : "+ str(self.decay))

    #self.random = Random(FLAGS.python_seed)

config = Config()

print("batch_size : "+ str(FLAGS.batch_size))


# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables


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

#run_with_config(1, trX, trY, teX, teY)

for learning_rate in [0.0001, 0.002]: #1, 0.0025, 0.002]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for decay in [0.9]: #[0.005, 0.01]:
    	config = Config()
    	config.learning_rate = learning_rate
    	config.decay = decay
       	run_with_config(config, trX, trY, teX, teY)
