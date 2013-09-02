---
template:inverse

background-image:url(pictures/KyotoFushimiInariLarge.jpg)

# 3 Finding your way

---
template:inverse

background-image:url(pictures/pollock.jpg)

# 3.1 Code organization

---

layout:false

.left-column[
  ## Main sub-packages
]

.right-column[
.red[numpy/core]: the meat of NumPy (focus of the tutorial):

- code for multiarray (src/multiarray), ufunc extensions (src/umath)

- various support libraries (npymath, npysort)

- public headers in include

Other important parts:

- .red[numpy/lib]: various tools on top of core

- .red[numpy/fft], .red[numpy/linalg], .red[numpy/random]

Other parts not on topic for this tutorial
]

---
layout:false

.left-column[
  ##npymath
]

.right-column[
.red[npymath] is a small C99 abstraction for cross platform math operations

- static library linked into the extensions that need it

- implement fundamental IEEE 754-related features (npy_isnan/npy_isinf/etc...)

- half float implementation

- C99 layer for functions, macros and constant definitions

npy_* functions should be used throughout numpy C code (e.g. npy_exp, not exp)

```
#include <numpy/npy_math.h>

void foo()
{
	double x = npy_exp(2.0);
}
```

API "documented" in numpy/npy_math.h header
]

---
layout:false

.left-column[
  ## Multiarray extension bird's eye view
]

.right-column[Contain the implementation of the array, dtype and iterators
objects:

- .red[PyArrayObject] struct (numpy/ndarraytypes.h): array object (hidden in recent versions)
- .red[PyArray_Descr] struct (ditto): dtype object
- .red[PyArrayMultiIterObject] struct (ditto): iterator object (used in broadcasting)

```
/* in numpy/ndarraytypes.h  */
struct PyArrayObject {
  char *data; /* Pointer to the raw data buffer */
  int nd; /* number of dimensions */
  npy_intp *dimensions; /* The size in each dimension */
  npy_intp *strides; /* strides array */
  PyObject *base;
  PyArray_Descr *descr; /* Pointer to type structure */
  int flags;
  PyObject *weakreflist;
}
```

One numpy array -> one PyArrayObject instance

]

---
layout:false

.left-column[
  ## dtype implementation details
]

.right-column[
PyArray_Descr contains the instance-specific data of a dtype

```c
/* in numpy/ndarraytypes.h  */
struct PyArray_Descr {
  ...
  char kind;
  ...
  /* used for structured array */
  struct _arr_descr *subarray;
  PyObject *fields;
  PyObject *names;
  /* a table of functions specific for each */
  PyArray_ArrFuncs *f;
  /* Metadata about this dtype */
  PyObject *metadata;
  NpyAuxData *c_metadata;
}
```

One dtype object -> one PyArray_Descr instance.

```c
/* each dtype has its own set of function pointers */
PyArray_ArrFuncs {
  ...
  PyArray_CopySwapNFunc *copyswapn;
  PyArray_CopySwapFunc *copyswap;
  ...
}
```

]

---
<!--layout:false-->
.left-column[
  ## PyArray_Type: your main ticket to follow code flow
]

.right-column[
.up30[
PyArrayType is an extension type (singleton) which defines the array behavior

```c
// in src/multiarray/arrayobject.c
// code simplified for presentation
PyTypeObject PyArray_Type = {
   ...
   array_repr, /* __repr__ */
   &array_as_number, /* number protocol */
   &array_as_sequence, /* sequence protocol */
   &array_as_mapping, /* mapping protocol */
   ...
   array_str, /* __str__ */
   &array_as_buffer, /* buffer protocol */
   ...
   array_iter, /* iter protocol */
   ...
   array_methods, /* array methods */
   ...
}
```

Critical to understand code flow of an operation, e.g.:

```
a = np.random.randn(100)
# How to find below op entry point in the C code ?
b = a + a
```

Addition is part of the number protocol -> look into array_as_number array.
]
]

---
layout:false

.left-column[
  ## Example
]

.right-column[
First, let`s compile numpy in debug mode:

```
# Create a virtualenv with debug python
virtualenv debug-env -p python2.7-dbg
source debug-env/bin/activate
# install bentomaker in that venv
cd $bento_sources && python setup.py install
# Build numpy in debug mode
CFLAGS="-O0 -g" LDFLAGS="-g" bentomaker build -i
```

We want to look into the snippet below to 'catch' the main entry point for '+'

```
# test_add.py
import numpy as np
a = np.array([1, 2, 3, 4])
b = a + a
```

In gdb

```
gdb python
> break array_add
> run test_add.py
> p ((int*)PyArray_DATA(m1))[0]
```

]

---
layout:false

.left-column[
.down50[
.down50[
.down50[
.rot90[
  ## PyArrayDescr_Type
]]]]]

.right-column[
PyArrayDescr_Type is an extension type (singleton) which defines the *dtype* class

```
/* in src/multiarray/descriptor.c */
/* code simplified for presentation */
PyTypeObject PyArrayDescr_Type = {
    ...
    "numpy.dtype",
    ...
    /* sequence protocol 
       used e.g. in structured dtype */
    &descr_as_sequence,
    /* mapping protocol 
       used e.g. in structured dtype */
    &descr_as_mapping,
    ...
    arraydescr_methods, /* methods */
    arraydescr_members, /* members */
    arraydescr_getsets, /* getset (properties) */
}
```

Less critical to understand Python <-> C layer.
]

<!--
---
template:inverse
background-image:url(pictures/confusing_2.jpg)

## What if you can`t find your way ?

---

.left-column[
  ## Poor man's callgraphs
]

.right-column[
Sometimes, you have no clue where to start (or just lazy):

- you can use dtrace to get calltrace at runtime (dapptrace in dtrace toolkit)

- poor man's replacement with Linux's perf tool:

```
# included in the USB key, also avail at 
# http://bit.ly/10OI2UD
import numpy as np
from minilib import under_perf

a = np.random.randn(10)

while True:
    with under_perf():
        a + a
```

```bash
$ python test.py
$ Ctrl+c
# you should now get a perf.data file
```
]

---

.left-column[
  ## Poor man's callgraphs (Cont.)
]

.right-column[
To look into the data:

```
$ perf -G -g --stdio
```

This will produce something like:

```
# condensed version
3.96%   python  libm-2.17.so
         |
         --- __libc_start_main
             main
    	 ...
             PyNumber_Add
             binary_op1
             array_add
             PyArray_GenericBinaryFunction
             _PyObject_CallFunction_SizeT
             call_function_tail
             PyObject_Call
             ufunc_generic_call
             PyUFunc_GenericFunction
	     ...
```

`
Caveat: sampling based (hence while True hack for short-lived functions).
Better support for dynamic probes in user space on the way.
`
]

---

.left-column[
  ## Practice
]

.right-column[
Try to fix https://github.com/numpy/numpy/issues/2592

> numpy.fromiter with a count argument hides any exception that might be raised
> by the iterator as: ValueError: iterator too short

```
# test_fromiter_bug.py
import numpy as np
def load_data(n):
    for e in xrange(n):
        if e == 42:
            raise Exception('42 is really a bad value')
        yield e

a = np.fromiter(load_data(50), dtype=int, count=50)
```

Questions:

- find out where C is called
- can you understand the bug ?
- fix it !
]
