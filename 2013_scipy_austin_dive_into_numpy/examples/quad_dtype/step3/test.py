import numpy as np

import _quad

a = np.array([1, 2, 3, 4], np.float)
b = a.astype(_quad.qdouble)

print b.dtype
print b

print b[0] + b[1]

# This will fail
#print b + b

