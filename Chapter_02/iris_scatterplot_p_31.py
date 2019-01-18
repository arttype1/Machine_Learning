#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                 header=None)
# select setosa  and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[:100, [0,2]].values

# train a perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_, marker= 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
