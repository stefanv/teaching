import numpy as np
import my_ufunc

x = np.arange(10)

print x, "\n"
print "square(x) ="
print my_ufunc.square(x)

y = np.arange(5)
y = y + y[:, None]

print "\n", y, "\n"
print "square(x) = "
print my_ufunc.square(y)
