#ifndef _PYQUAD_H_
#define _PYQUAD_H_

#include "common.h"

int PyQuad_Check(PyObject* object);
qdouble PyQuad_AsQuad(PyObject* p);
PyObject* PyQuad_FromQuad(qdouble x);

typedef struct {
        PyObject_HEAD
        qdouble obval;
} PyQuad;

PyTypeObject PyQuad_Type;

int
init_quad_type(PyObject *m, PyTypeObject* tp);

#endif
