from HAR_data_handler import get_HAR_data
import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
from base_config import Config, YaxisBundle, PlotUtil



def single_LSTM_cell(input_hidden_tensor, n_outputs):
    """ define the basic LSTM layer
        argument:
            input_hidden_tensor: list a list of tensor,
                                 shape: time_steps*[batch_size,n_inputs]
            n_outputs: int num of LSTM layer output
        return:
            outputs: list a time_steps list of tensor,
                     shape: time_steps*[batch_size,n_outputs]
    """
    with tf.variable_scope("lstm_cell"):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_outputs, state_is_tuple=True, forget_bias=0.999)
        #outputs, _ = tf.nn.rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
    return outputs




def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # Exchange dim 1 and dim 0
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # New feature_mat's shape: [time_steps, batch_size, n_inputs]

    # Temporarily crush the feature_mat's dimensions
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # New feature_mat's shape: [time_steps*batch_size, n_inputs]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    hidden = tf.split(axis=0, num_or_size_splits=config.n_steps, value=feature_mat)
    print (len(hidden), str(hidden[0].get_shape()))

    outputs = single_LSTM_cell(hidden, config.n_hidden)


    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']



################################## load data and config ##################################

X_train, y_train, X_test, y_test = get_HAR_data()


class SingleLayerConfig(Config):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        super(SingleLayerConfig, self).__init__()
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # == 9 Features count is of 9: three 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes

        self.tensor_board_logging_enabled = True
        self.logs_path = "/tmp/LSTM_logs/single_layer_lstm/"
        self.tensorboard_cmd = "tensorboard --logdir="+ self.logs_path



#config = Config(X_train, X_test)
config = SingleLayerConfig()


def run_with_config(config) : #, X_train, y_train, X_test, y_test):
    tf.reset_default_graph()  # To enable to run multiple things in a loop
    config.print_config()


    config.W = {
        'hidden': tf.Variable(tf.random_normal([config.n_inputs, config.n_hidden])),
        'output': tf.Variable(tf.random_normal([config.n_hidden, config.n_classes]))
    }
    config.biases = {
        'hidden': tf.Variable(tf.random_normal([config.n_hidden], mean=1.0)),
        'output': tf.Variable(tf.random_normal([config.n_classes]))
    }

    #-----------------------------------
    # Define parameters for model
    #-----------------------------------


    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    print "Unregularised variables:"
    for unreg in [tf_var.name for tf_var in tf.trainable_variables() if
                  ("noreg" in tf_var.name or "Bias" in tf_var.name)]:
        print unreg

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    # ------------------------------------------------------
    # step3.5 : Tensorboard stuff here
    # ------------------------------------------------------
    if config.tensor_board_logging_enabled:
        tf.summary.scalar("loss", cost)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()


    # --------------------------------------------
    # step4: Hooray, now train the neural network
    # --------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run()

    if config.tensor_board_logging_enabled:
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(config.logs_path, graph=tf.get_default_graph())

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            if config.tensor_board_logging_enabled :
                _, summary = sess.run([optimizer, merged_summary_op], feed_dict={X: X_train[start:end],Y: y_train[start:end]})
            else:
                sess.run(optimizer, feed_dict={X: X_train[start:end],Y: y_train[start:end]})


        if config.tensor_board_logging_enabled:
            # Write logs at every iteration
            summary_writer.add_summary(summary, i)

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
            X: X_test, Y: y_test})
        print("traing iter: {},".format(i) + \
              " test accuracy : {},".format(accuracy_out) + \
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    if config.tensor_board_logging_enabled:
        print("Run the command line:\n")
        print(config.tensorboard_cmd)
        print("\nThen open http://0.0.0.0:6006/ into your web browser")

if __name__ == '__main__':
    run_with_config(config) #, trX, trY, teX, teY)


