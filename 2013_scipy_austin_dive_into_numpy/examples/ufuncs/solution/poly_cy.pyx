cimport numpy as cnp
import numpy as np

def fast_poly(double[:] x, double a):
    cdef double[:] out = np.empty(shape=x.shape[0], dtype=np.float64)
    cdef int i
    for i in range(x.shape[0]):
        out[i] = (x[i] * x[i] + 3) - a * x[i] + 5
    return np.asarray(out)
