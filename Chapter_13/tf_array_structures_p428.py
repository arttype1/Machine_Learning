#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
#
import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6), name='x2')
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')
with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)
    print('input shape:', x_array.shape)
    print('Reshaped:\n', sess.run(x2, feed_dict={x: x_array}))
    print('Column Sums:\n', sess.run(xsum, feed_dict={x: x_array}))
    print('Column Means:\n', sess.run(xmean, feed_dict={x: x_array}))

