import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 2, 3])  # 1-D tensor
y = tf.slice(x, [1, 0,0], [1,1,2])

#initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#run
result = sess.run(y, feed_dict={x:[[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]})
print(result)