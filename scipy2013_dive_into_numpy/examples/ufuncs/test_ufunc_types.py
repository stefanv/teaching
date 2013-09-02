import numpy as np
import my_ufunc_types

x = np.array([1.0, 2.0, 3.0, 4.0])

for dtype in (np.int8, np.int32, np.int64, np.float32, np.float64):
    print "square(x) [dtype=%s]" % dtype
    print my_ufunc_types.square(x.astype(dtype))
    print

# Bench: specialized loop vs cast loop
import timeit

from my_ufunc import square as square_double
from my_ufunc_types import square as square_all


t0 = timeit.timeit('square_double(x)',
                   setup="from __main__ import square_double, x",
                   number=1000)

t1 = timeit.timeit('square_all(x)',
                   setup="from __main__ import square_all, x",
                   number=1000)

print "Time decrease by specialization = %.0f%%" % (100 * (t1 / t0 - 1))
