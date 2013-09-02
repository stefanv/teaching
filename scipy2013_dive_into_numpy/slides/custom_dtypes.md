---
template:inverse

# 5. Writing a custom dtype

---
.left-column[
## Synopsis
]
.right-column[
We will create the skeleton for quad precision dtype:

- IEEE-754 2008 float128 standard

- available in recent gcc with __float128

Creating a new dtype involves:

- wrapping the basic type as a python object (no numpy support), implement number protocol

- creating a new dtype on top of it

- add some basic ufunc for it
]

---

.left-column[
## A few words on quadruple precision
]

.right-column[
Specified in the recent IEEE-754 standard:

- few CPU support it natively

- 113 effective bits of precision (~36 decimal digits)

- gcc >= 4.6 + libquadmath add software support for it for Intel architectures

Example

```c
#include <quadmath.h>

int main()
{
	__float128 a = 1.0;
	char str[1024];
	int st;

	a *= 2.3;

	st = quadmath_snprintf(str, sizeof(str), "%Qf", q);
	if (st < 0) {
		printf("Error while converting to string\n");
	}
}
```
]
---
template:inverse
## 5.1 Wrapping a float128 into a python object 

---
.left-column[
## A simple quad object
]
.right-column[
The smallest thing we need:

- create a qdouble object type
- allow to create one (from a string, a double, etc...)
- allow to print it

```
typedef __float128 qdouble;
/* boxed qdouble */
typedef struct {
        PyObject_HEAD
        qdouble obval;
} PyQuad;

static PyTypeObject PyQuad_Type = {
...
pyquad_repr,
...
pyquad_new,
}

PyMODINIT_FUNC init_quad(void) {
    ...
    if (PyType_Ready(&PyQuad_Type) < 0) {
        return;
    }
    Py_INCREF(&PyQuad_Type);
    PyModule_AddObject(m, "qdouble",
		(PyObject*)&PyQuad_Type);
}
```

]

---
.left-column[
## Exercise
]

.right-column[
See usb/material/examples/quad_dtype/step1:

- compile the extension module

- confirms you can create a quad object, and display it

- add new features such as hashing support, more types for ctor, or comparison
]
---

.left-column[
## Number protocol
]
.right-column[
Arithmetic operations are supported through the number protocol.

```c
PyObject*
pyquad_add(PyObject *a, PyObject *b)
{
   qdouble x, y;
   x = PyQuad_AsQuad(a);
   y = PyQuad_AsQuad(b);
   return PyQuad_FromQuad(x + y);
}

PyNumberMethods pyquad_as_number {
    pyquad_add,          /* nb_add */
    ...
}
```

This can then be registered into PyQuad_Type tb_as_number:

```c
PyTypeObject PyQuad_Type = {
    ..
    0,
    pyquad_repr,
    &pyquad_as_number,
    0,
}
```
]
---
.left-column[
## Exercise
]

.right-column[
See usb/material/examples/quad_dtype/step2:

- compile the extension module

- confirms you can add 2 quad objects

- add a few more operations
]

---
template:inverse
## 5.2 Wrapping a python quad into a dtype

---

.left-column[
## Registering a new dtype
]
.right-column[
The fundamental structure is npyquad_descr:

```c
PyArray_ArrFuncs NpyQuad_ArrFuncs;

typedef struct { char c; qdouble q; } align_test;

PyArray_Descr npyquad_descr = {
   PyObject_HEAD_INIT(0)
   &PyQuad_Type, /* typeobj */
   'f', /* kind (matters for coercion) */
   'q', /* type */
   '=', /* byteorder */
   NPY_NEEDS_PYAPI | NPY_USE_GETITEM | \
	NPY_USE_SETITEM, /* hasobject */
   0,                      /* type_num */
   sizeof(qdouble),       /* elsize */
   offsetof(align_test, q), /* alignment */
   ...
   &NpyQuad_ArrFuncs,  /* f */
}
```

Registered as followed:

```
npyquad_descr.ob_type = &PyArrayDescr_Type;
npy_registered_quadnum = \
	PyArray_RegisterDataType(&npyquad_descr);
```
]
---

.left-column[
## Registering a new dtype
]
.right-column[
NpyQuad_ArrFuncs is an array of function pointers doing most of the
implementation:

```
/* pick up one item from data buffer into a scalar */
static PyObject *
npyquad_getitem(char *data, PyArrayObject arr)
{
    qdouble q;

    memcpy(&q, data, sizeof(q));
    return PyQuad_FromQuad(q);
}

init_quad_descriptor(PyObject *np_module)
{
    PyArray_InitArrFuncs(&NpyQuad_ArrFuncs);
    NpyQuad_ArrFuncs.getitem = \
	(PyArray_GetItemFunc*)npyquad_getitem;
    NpyQuad_ArrFuncs.setitem = \
	(PyArray_SetItemFunc*)npyquad_setitem;
    NpyQuad_ArrFuncs.copyswap = \
	(PyArray_CopySwapFunc*)npyquad_copyswap;

    ...
}
```
]

---
.left-column[
## Example of usage
]
.right-column[
In material/quad_dtype/step3, example of usage:

```python
import numpy as np

import _quad

a = np.array([1, 2, 3, 4], np.float)
b = a.astype(_quad.qdouble)

print b[0] + b[1]
```

But the following fails
```
a = np.array([1, 2, 3, 4], _quad.qdouble)
```

Can you fix it ?
]
---
.left-column[
## Adding ufunc
]
.right-column[
Conceptually, a ufunc needs at least the following:

- a core loop that is given a 1d buffer (but not contiguous)

- to be registered

```c
void
quad_ufunc_add(char** args, npy_intp* dimensions,
	       npy_intp* steps, void* data);

int
register_ufuncs(PyObject* np_module,
	        int npy_registered_quadnum)
{
    PyUFuncObject* ufunc = (PyUFuncObject*)
		PyObject_GetAttrString(np_module, "add");
    int args[3] = {npy_registered_quadnum,
		   npy_registered_quadnum,
		   npy_registered_quadnum};

    ...
    if (PyUFunc_RegisterLoopForType(ufunc,
		 npy_registered_quadnum,
		 quad_ufunc_add, args, 0) < 0) {
        return -1;
    }

    return 0;
}
```
]

---
.left-column[
## Exercise
]
.right-column[
Try to add a few more ufuncs
]
---
template:inverse
