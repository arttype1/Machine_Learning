 #code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))


z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y =1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y =0')
plt.ylim(0.0, 5.1)
plt.ylabel('J(w)')
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.legend(loc='upper center')
plt.show()
