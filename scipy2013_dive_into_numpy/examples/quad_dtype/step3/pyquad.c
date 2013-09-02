#include "pyquad.h"
#include "number.h"

int
PyQuad_Check(PyObject* object)
{
    return PyObject_IsInstance(object, (PyObject*)&PyQuad_Type);
}

PyObject*
PyQuad_FromQuad(qdouble x)
{
    PyQuad* p = (PyQuad*)PyQuad_Type.tp_alloc(&PyQuad_Type, 0);
    if (p) {
        p->obval = x;
    }
    return (PyObject*)p;
}

PyObject *
pyquad_new(PyTypeObject *NPY_UNUSED(type), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    qdouble q;
    char *c_literal, *remain;

    if (!PyArg_ParseTuple(args, "s", &c_literal))
        return NULL;

    q = strtoflt128(c_literal, &remain);
    return PyQuad_FromQuad(q);
}

PyObject *
pyquad_str(PyObject *self)
{
    char str[128];
    int st;
    qdouble q = ((PyQuad*)self)->obval;

    st = quadmath_snprintf(str, sizeof(str), "%Qf", q);
    if (st < 0) {
        fprintf(stderr, "BAD\n");
    }
    return PyString_FromString(str);
}

PyObject *
pyquad_repr(PyObject *self)
{
    char str[128];
    int st;
    qdouble q = ((PyQuad*)self)->obval;

    st = quadmath_snprintf(str, sizeof(str), "%#.*Qf", FLT128_DIG, q);
    if (st < 0) {
        fprintf(stderr, "BAD\n");
    }
    return PyString_FromString(str);
}

PyTypeObject PyQuad_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
    "quad",                                   /* tp_name */
    sizeof(PyQuad),                           /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    pyquad_repr,                                        /* tp_repr */
    &pyquad_as_number,                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    pyquad_str,
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /* tp_flags */
    "Quad precision floating numbers",        /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    0,                           /* tp_new */
    0,                                        /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

int
init_quad_type(PyObject *m, PyTypeObject* tp)
{
    PyQuad_Type.tp_base = tp;
    PyQuad_Type.tp_new = pyquad_new;
    if (PyType_Ready(&PyQuad_Type) < 0) {
        return;
    }

    Py_INCREF(&PyQuad_Type);
    PyModule_AddObject(m, "qdouble", (PyObject*)&PyQuad_Type);
}
