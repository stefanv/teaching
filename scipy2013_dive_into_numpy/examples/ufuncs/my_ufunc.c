#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>


static void square_d(char** args,
                     npy_intp* dimensions, npy_intp* steps,
                     void *NPY_UNUSED(data))
{
    npy_intp n = dimensions[0];

    char *input = args[0];
    char *output = args[1];

    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    double tmp;
    npy_intp i;

    for (i = 0; i < n; i++) {
        tmp = *( (double *)input );
        tmp = tmp * tmp;

        *((double *)output) = tmp;

        input += in_step;
        output += out_step;
    }
}

static PyUFuncGenericFunction funcs[1] = {&square_d};
static void *data[1] = {NULL};
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};


/* -------------------------------------------------------------------------- */
/* See http://docs.python.org/2.7/extending/index.html                        */
/* -------------------------------------------------------------------------- */

PyMODINIT_FUNC initmy_ufunc(void)
{
    PyObject *m, *ufunc, *d;

    /* Start with no methods, will add after initializing the ufunc */
    static PyMethodDef module_methods[] = {
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

    m = Py_InitModule("my_ufunc", module_methods);
    if (m == NULL) return;

    /* Initialize numpy structures */
    import_array();
    import_umath();

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
