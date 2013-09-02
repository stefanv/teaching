import numpy as np

import _quad

a = np.array([1, 2, 3, 4], np.float).astype(_quad.qdouble)
b = np.array([1, 2, 3, 4], np.float).astype(np.float)

c = a + b
print c
