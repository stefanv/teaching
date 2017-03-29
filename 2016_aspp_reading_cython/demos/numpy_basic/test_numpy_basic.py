from numpy_basic import foo
import numpy as np

x = np.arange(24.0).reshape((8, 3))
print("Before:")
print(x)

y = foo(x)

print("After:")
print(y)
