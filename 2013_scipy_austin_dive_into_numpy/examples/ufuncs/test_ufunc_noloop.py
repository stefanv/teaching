import numpy as np
import my_ufunc_noloop

x = np.arange(10).astype(np.double)

print x, "\n"
print "square(x) ="
print my_ufunc_noloop.square(x)
