#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import os
import struct
import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

X_train, y_train = load_mnist('mnist', kind='train')
# print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
nn.fit(X_train=X_train[:55000], y_train=y_train[:55000], X_valid=X_train[55000:], y_valid=y_train[55000:])
print('done!')


"""
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax =  ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""


