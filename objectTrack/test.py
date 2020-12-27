import numpy as np

a = [-1, -2]
a = np.asarray(a)
b = a<0
if any(a<0):
    print('yes')
