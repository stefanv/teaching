#include <quadmath.h>

#include "common.h"
#include "number_protocol.h"

typedef struct {
        PyObject_HEAD
        qdouble obval;
} PYQUAD_PRIVATE;

/* Declare the singleton that will hold the type information for a quad double
 */
PyTypeObject PyQuad_Type;

/* Create the given python object is a quad object */
int
PyQuad_Check(PyObject* object)
{
    return PyObject_IsInstance(object, (PyObject*)&PyQuad_Type);
}

qdouble
PyQuad_AsQuad(PyObject* p)
{
	PYQUAD_PRIVATE *pp;
        if(PyQuad_Check(p)) {
            pp = (PYQUAD_PRIVATE*)p;
            return pp->obval;
        } else {
            fprintf(stderr, "Not Implemented"); 
            exit(EXIT_FAILURE);
        }
}

/* Create a python quad from a C quad */
PyObject*
PyQuad_FromQuad(qdouble x)
{
    PYQUAD_PRIVATE* p = (PYQUAD_PRIVATE*)PyQuad_Type.tp_alloc(&PyQuad_Type, 0);
    if (p) {
        p->obval = x;
    }
    return (PyObject*)p;
}

/*
 * Functions of the extension type
 */
static PyObject *
pyquad_new(PyTypeObject *NPY_UNUSED(type), PyObject *args,
	   PyObject *NPY_UNUSED(kwds))
{
    qdouble q;
    char *c_literal, *remain;

    if (!PyArg_ParseTuple(args, "s", &c_literal))
        return NULL;

    q = strtoflt128(c_literal, &remain);
    return PyQuad_FromQuad(q);
}

static PyObject *
pyquad_str(PyObject *self)
{
    char str[128];
    int st;
    qdouble q = PyQuad_AsQuad(self);

    st = quadmath_snprintf(str, sizeof(str), "%Qf", q);
    if (st < 0) {
        fprintf(stderr, "BAD\n");
    }
    return PyString_FromString(str);
}

static PyObject *
pyquad_repr(PyObject *self)
{
    char str[128];
    int st;
    qdouble q = PyQuad_AsQuad(self);

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
    sizeof(PYQUAD_PRIVATE),                   /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    pyquad_repr,                              /* tp_repr */
    &pyquad_as_number,                         /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    pyquad_str,                               /* tp_str */
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
    0,        			              /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    pyquad_new,                               /* tp_new */
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
