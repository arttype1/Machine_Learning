#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
#
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

# create a graph
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name='X')
    w = tf.Variable(2.0, name='weight')
    b = tf. Variable(0.7, name='bias')
    z = w*x+b
    init = tf.global_variables_initializer()

# create a session and pass in graph g
with tf.Session(graph=g) as sess:
    ## initialize w and b:
    sess.run(init)
    ## evaluate z:
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z= %4.1f' %(t, sess.run(z, feed_dict={x:t})))
