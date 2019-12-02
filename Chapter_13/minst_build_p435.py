#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
#
import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
from tflinreg import TfLinreg
from load_mnist import load_mnist

X_train,y_train = load_mnist('./mnist/', kind='train')
print('Rows: %d, Columns: %d' %(X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('./mnist', kind='t10k')
print('Rows: %d, Columns: %d' %(X_test.shape[0], X_test.shape[1]))
# mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test

n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name = 'tf_x')

    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name = 'tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name='layer2')
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name='layer3')

predictions = {
    'classes': tf.argmax(logits, axis=1, name='predicted_classes'),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
}

# define cost function and optimizer:
with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)
    init_op = tf.global_variables_initializer()

def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].asttype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])


# create a session to launch the graph
sess = tf.Session(graph=g)
sess.run(init_op)
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)
    for batch_X, batch_y, in batch_generator:
        ## prepare a dict to feed data to our network:
        feed = {tf_x:batch_X, tf_y:batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(' -- Epoch %d Avg. Traing Loss: %4f' % (epoch+1, np.mean(training_costs)))

# do prediction on the test set:
feed = {tf_x : X_test_centered}
y_pred = sess.run(predictions['classes'], feed_dict=feed)

print('Test Accuracy: %.2f%%' % (100*np.sum(y_pred == y_test) / y_test.shape[0]))
