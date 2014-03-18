from numpy_ufunc_api cimport *
from numpy cimport *

import_ufunc() # must come first!!!!

DEF _UFUNCS_MAX = 100
DEF _UFUNCS_S_MAX = 300

cdef:
    PyUFuncGenericFunction _ufuncs_functions[_UFUNCS_MAX]
    void* _ufuncs_data[_UFUNCS_MAX]
    char _ufuncs_signatures[_UFUNCS_S_MAX]
    int _ufuncs_set = 0
    int _ufuncs_s_set = 0

cdef _ufuncs_add_def(void *data, PyUFuncGenericFunction function,
        char signature, int num_signatures):
    cdef int i
    global _ufuncs_set, _ufuncs_s_set
    _ufuncs_functions[_ufuncs_set] = function
    _ufuncs_data[_ufuncs_set] = data
    _ufuncs_set += 1
    for i in range(num_signatures):
        _ufuncs_signatures[_ufuncs_s_set] = signature
        _ufuncs_s_set += 1


cdef register_ufunc_d(double d(double), char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>d, PyUFunc_f_f_As_d_d, NPY_FLOAT, 2)
    _ufuncs_add_def(<void*>d, PyUFunc_d_d, NPY_DOUBLE, 2)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 2, 1, 1,
                identity, name, doc, 0)

cdef register_ufunc_fd(float f(float), double d(double),
        char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>f, PyUFunc_f_f, NPY_FLOAT, 2)
    _ufuncs_add_def(<void*>d, PyUFunc_d_d, NPY_DOUBLE, 2)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 2, 1, 1,
                identity, name, doc, 0)

cdef register_ufunc_fdg(float f(float), double d(double),
        long double g(long double), char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>f, PyUFunc_f_f, NPY_FLOAT, 2)
    _ufuncs_add_def(<void*>d, PyUFunc_d_d, NPY_DOUBLE, 2)
    _ufuncs_add_def(<void*>g, PyUFunc_g_g, NPY_LONGDOUBLE, 2)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 3, 1, 1,
                identity, name, doc, 0)

cdef register_ufunc_dd(double dd(double, double),
        char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>dd, PyUFunc_ff_f_As_dd_d, NPY_FLOAT, 3)
    _ufuncs_add_def(<void*>dd, PyUFunc_dd_d, NPY_DOUBLE, 3)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 2, 2, 1,
                identity, name, doc, 0)

cdef register_ufunc_ffdd(float ff(float, float), double dd(double, double),
        char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>ff, PyUFunc_ff_f, NPY_FLOAT, 3)
    _ufuncs_add_def(<void*>dd, PyUFunc_dd_d, NPY_DOUBLE, 3)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 2, 2, 1,
                identity, name, doc, 0)

cdef register_ufunc_ffddgg(float ff(float, float), double dd(double, double),
        long double gg(long double, long double),
        char *name, char *doc, int identity):
    cdef:
        PyUFuncGenericFunction *functions = &_ufuncs_functions[_ufuncs_set]
        void* *data = &_ufuncs_data[_ufuncs_set]
        char *signatures = &_ufuncs_signatures[_ufuncs_s_set]
    _ufuncs_add_def(<void*>ff, PyUFunc_ff_f, NPY_FLOAT, 3)
    _ufuncs_add_def(<void*>dd, PyUFunc_dd_d, NPY_DOUBLE, 3)
    _ufuncs_add_def(<void*>gg, PyUFunc_gg_g, NPY_LONGDOUBLE, 3)
    return PyUFunc_FromFuncAndData(functions, data, signatures, 3, 2, 1,
                identity, name, doc, 0)

