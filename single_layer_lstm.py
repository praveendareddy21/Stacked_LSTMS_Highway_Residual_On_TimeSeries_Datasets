from HAR_data_handler import get_HAR_data
import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np

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

    # Linear activation, reshaping inputs to the LSTM's number of hidden:
    hidden = tf.nn.relu(
        tf.matmul(feature_mat, config.W['hidden'])
        + config.biases['hidden'])
    # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    hidden = tf.split(axis=0, num_or_size_splits=config.n_steps, value=hidden)
    # New hidden's shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

    outputs = single_LSTM_cell(hidden, config.n_hidden)


    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']



def run_with_config(config) : #, X_train, y_train, X_test, y_test):
    tf.reset_default_graph()  # To enable to run multiple things in a loop

    #-----------------------------------
    # Define parameters for model
    #-----------------------------------

    X_train, y_train, X_test, y_test = get_HAR_data()
    #config = Config(X_train, X_test)
    config.n_inputs = len(X_train[0][0])
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # Let's get serious and build the neural network
    #------------------------------------------------------
    with tf.device("/cpu:0"):  # Remove this line to use GPU. If you have a too small GPU, it crashes.
        X = tf.placeholder(tf.float32, [
                           None, config.n_steps, config.n_inputs], name="X")
        Y = tf.placeholder(tf.float32, [
                           None, config.n_classes], name="Y")

        # is_train for dropout control:
        is_train = tf.placeholder(tf.bool, name="is_train")
        keep_prob_for_dropout = tf.cond(is_train,
            lambda: tf.constant(
                config.keep_prob_for_dropout,
                name="keep_prob_for_dropout"
            ),
            lambda: tf.constant(
                1.0,
                name="keep_prob_for_dropout"
            )
        )

        pred_y = LSTM_Network(X, config)

        # Loss, optimizer, evaluation

        # Softmax loss with L2 and L1 layer-wise regularisation
        print "Unregularised variables:"
        for unreg in [tf_var.name for tf_var in tf.trainable_variables() if ("noreg" in tf_var.name or "Bias" in tf_var.name)]:
            print unreg
        l2 = config.lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        # first_weights = [w for w in tf.all_variables() if w.name == 'LSTM_network/layer_1/pass_forward/relu_fc_weights:0'][0]
        # l1 = config.lambda_loss_amount * tf.reduce_mean(tf.abs(first_weights))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=Y)) + l2  # + l1

        # Gradient clipping Adam optimizer with gradient noise
        optimize = tf.contrib.layers.optimize_loss(
            loss,
            global_step=tf.Variable(0),
            learning_rate=config.learning_rate,
            optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
            clip_gradients=config.clip_gradients,
            gradient_noise_scale=config.gradient_noise_scale
        )

        correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.

    sessconfig = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=sessconfig) as sess:
        tf.global_variables_initializer().run()

        best_accuracy = (0.0, "iter: -1")
        best_f1_score = (0.0, "iter: -1")

        # Start training for each batch and loop epochs

        worst_batches = []

        for i in range(config.training_epochs):

            # Loop batches for an epoch:
            shuffled_X, shuffled_y = shuffle(X_train, y_train, random_state=i*42)
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):

                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: shuffled_X[start:end],
                        Y: shuffled_y[start:end],
                        is_train: True
                    }
                )

                worst_batches.append(
                    (train_loss, shuffled_X[start:end], shuffled_y[start:end])
                )
                worst_batches = list(sorted(worst_batches))[-5:]  # Keep 5 poorest

            # Train F1 score is not on boosting
            train_f1_score = metrics.f1_score(
                shuffled_y[start:end].argmax(1), train_pred.argmax(1), average="weighted"
            )

            # Retrain on top worst batches of this epoch (boosting):
            # a.k.a. "focus on the hardest exercises while training":
            for _, x_, y_ in worst_batches:

                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: x_,
                        Y: y_,
                        is_train: True
                    }
                )

            # Test completely at the end of every epoch:
            # Calculate accuracy and F1 score
            pred_out, accuracy_out, loss_out = sess.run(
                [pred_y, accuracy, loss],
                feed_dict={
                    X: X_test,
                    Y: y_test,
                    is_train: False
                }
            )

            # "y_test.argmax(1)": could be optimised by being computed once...
            f1_score_out = metrics.f1_score(
                y_test.argmax(1), pred_out.argmax(1), average="weighted"
            )

            print (
                "iter: {}, ".format(i) + \
                "train loss: {}, ".format(train_loss) + \
                "train accuracy: {}, ".format(train_acc) + \
                "train F1-score: {}, ".format(train_f1_score) + \
                "test loss: {}, ".format(loss_out) + \
                "test accuracy: {}, ".format(accuracy_out) + \
                "test F1-score: {}".format(f1_score_out)
            )

            best_accuracy = max(best_accuracy, (accuracy_out, "iter: {}".format(i)))
            best_f1_score = max(best_f1_score, (f1_score_out, "iter: {}".format(i)))

        print("")
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("final F1 score: {}".format(f1_score_out))
        print("best epoch's F1 score: {}".format(best_f1_score))
        print("")

    # returning both final and bests accuracies and f1 scores.
    return accuracy_out, best_accuracy, f1_score_out, best_f1_score
