#ifndef _PYTHON_WRAPPER_H
#define _PYTHON_WRAPPER_H

#include "common.h"

typedef struct PyQuad_tag PyQuad;

extern PyTypeObject PyQuad_Type;

int PyQuad_Check(PyObject* object);

qdouble PyQuad_AsQuad(PyObject* p);
PyObject* PyQuad_FromQuad(qdouble x);

#endif
