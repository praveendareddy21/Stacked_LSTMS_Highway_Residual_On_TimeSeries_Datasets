import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 2]) # 1-D tensor
y = tf.slice(x, [1, 1], [1,2])

#initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#run
result = sess.run(y, feed_dict={x: [[0,1], [0,2], [0,3], [0,4], [0,5]] })
print(result)


