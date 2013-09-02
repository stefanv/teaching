#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

double square_d(double d) {
    return d * d;
}

static void *data[] = {&square_d};
static char types[] = {NPY_DOUBLE, NPY_DOUBLE};
static PyUFuncGenericFunction funcs[sizeof(data)];



/* -------------------------------------------------------------------------- */
/* See http://docs.python.org/2.7/extending/index.html                        */
/* -------------------------------------------------------------------------- */

PyMODINIT_FUNC initmy_ufunc_noloop(void)
{
    PyObject *m, *ufunc, *d;

    /* Start with no methods, will add after initializing the ufunc */
    static PyMethodDef module_methods[] = {
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

    m = Py_InitModule("my_ufunc_noloop", module_methods);
    if (m == NULL) return;

    /* Initialize numpy structures */
    import_array();
    import_umath();

    funcs[0] = &PyUFunc_d_d;
    ufunc = PyUFunc_FromFuncAndData(funcs, data, types,
                                    1, /* nr of data types */
                                    1, 1, /* nr in & out */
                                    PyUFunc_None, /* identity */
                                    "square", /* function name */
                                    "docstring", /* docstring */
                                    0); /* unused */

    /* Add ufunc to module */
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "square", ufunc);
    Py_DECREF(ufunc);
}
