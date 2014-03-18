include "numpy_ufuncs.pxi"

cdef extern from "math.h":
    double lgammad "lgamma" (double)
    float lgammaf(float)
    long double lgammal(long double)

lgamma = register_ufunc_fdg(lgammaf, lgammad, lgammal,
    "lgamma", "log gamma function", PyUFunc_None)

