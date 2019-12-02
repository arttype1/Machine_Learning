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

import tensorflow.contrib.keras as keras
np.random.seed(123)
tf.set_random_seed(123)
y_train_onehot = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=64, epochs=50, verbose=0,
                    validation_split=0.1)
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))
