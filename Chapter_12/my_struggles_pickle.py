#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import os
import struct
import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
import pickle

X_train, y_train = load_mnist('mnist', kind='train')
# print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

with open('minst.nn', 'rb') as f:
    # unPickle the NN using the highest protocol available.
    nn = pickle.load(f)
y_test_pred = nn.predict(X_test)
miscl_img = X_test[y_test != y_test_pred][25:50]
correct_lab = y_test[y_test != y_test_pred][25:50]
miscl_lab = y_test_pred[y_test != y_test_pred][25:50]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


