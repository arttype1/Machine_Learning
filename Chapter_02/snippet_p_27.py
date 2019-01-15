#code is from 'Python Maching Learning' by Raschka and Mirialili
#################################################################################
import numpy as np

v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
angle = np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
print(f"Angle between v1 and v2 = {angle}")
