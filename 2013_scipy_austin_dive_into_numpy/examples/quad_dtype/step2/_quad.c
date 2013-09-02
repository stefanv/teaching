#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <quadmath.h>

#include "common.h"
#include "python_wrapper.h"
#include "number_protocol.h"

static PyMethodDef ModuleMethods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_quad(void)
{
    PyObject *m;

    m = Py_InitModule("_quad", ModuleMethods);
    if (m == NULL) {
        return;
    }

    if (PyType_Ready(&PyQuad_Type) < 0) {
        return;
    }
    Py_INCREF(&PyQuad_Type);
    PyModule_AddObject(m, "qdouble", (PyObject*)&PyQuad_Type);
}
