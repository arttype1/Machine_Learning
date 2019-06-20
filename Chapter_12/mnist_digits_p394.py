#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import os
import struct
import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt

X_train, y_train = load_mnist('mnist', kind='train')
# print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax =  ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


