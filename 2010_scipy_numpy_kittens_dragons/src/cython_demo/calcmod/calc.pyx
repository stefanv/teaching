cimport numpy as np

def compute_arr(np.ndarray[np.int_t, ndim=2] image):
    """Fill each element of the input array with i +  j.

    """
    cdef int i, j

    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]

    for i in range(rows):
        for j in range(cols):
            image[i, j] = i + j

cdef c_func_low_overhead(np.ndarray arr):
    cdef np.ndarray[np.int_t, ndim=1] x = arr
    cdef int i

    for i in range(x.shape[0]):
        x[i] = i

def cdef_frontend(np.ndarray[np.int_t, ndim=1] x):
    """
    Show how to pass ndarrays to cdef funcs.

    Takes 1D int array as input. Fills that array with arange(len(x)).

    """
    y = x.copy()
    c_func_low_overhead(y)
    return y

