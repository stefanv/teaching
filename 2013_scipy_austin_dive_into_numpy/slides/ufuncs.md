---
template: inverse

# 4. Universal Functions

---

.left-column[
  ## API
]

.right-column[
.up30[
``numpy/core/include/numpy/ufuncobject.h``

- Vectorized function that takes a fixed number of scalar inputs and produces a
fixed number of scalar outputs.

- Supports array broadcasting, type casting, and other standard features

Pure Python ufunc:

```python
def range_sum(a, b):
    return np.arange(a, b).sum()

rs = np.frompyfunc(range_sum, 2, 1)

x = np.array([[1, 2, 3, 4]])

>>> rs(x, x + 1)
array([[1, 2, 3, 4]], dtype=object)

>>> rs(x, x.T)
array([[0, 0, 0, 0],
       [1, 0, 0, 0],
       [3, 2, 0, 0],
       [6, 5, 3, 0]], dtype=object)
```

- Note that broadcasting is supported
- Unfortunately, the output is always an object array.  Slow, because
  we're wrapping a Python call.
]
]

---

.left-column[
  ## API: Calling
]

.right-column[

Keywords:

- ``out`` : Output array, useful for in-place computation.
- ``where`` : Ufunc only calculated where ``broadcast(mask, inputs) == True``.
  Careful! Undefined where no elements are encountered.
- ``casting`` : Casting behavior (more later).
- ``order`` : Calculation iteration order and memory layout of output array.
- ``dtype`` : Output *and calculation* dtype.  Often important for
  accumulators.
- ``sig`` : Data-type or signature string; indicates which underlying 1-D loop
  is executed (typically the loops are found automatically).  See ``types``
  attribute.  Example: ``ff->f``.
- ``extobj`` : Specify ufunc buffer size, error mode integer, error call-back
  function.

]

---

.left-column[
  ## Ufunc output type
]

.right-column[

- Determined by input class with highest `__array_priority__`.

```
class StrongArray(np.ndarray):
    __array_priority__ = 10


class WeakArray(np.ndarray):
    __array_priority__ = 1


>>> s = StrongArray([2, 2]); w = WeakArray([2, 2])

>>> type(s + w).__name__
StrongArray
```
]

---

.left-column[
  ## Ufunc output type (continued)
]

.right-column[
.up20[
Otherwise, by ``output`` parameter.  Output class may have following methods:

- ``__array_prepare__`` :
  Called before ufunc. Knows some context about the ufunc, and may be used to
  add, e.g., meta-data.  Output then passed to ufunc.

- ``__array_wrap__`` : Called after execution of ufunc.

```python
In [159]: class MyArray(np.ndarray):
     ...:     def __array_prepare__(self,
                     array, (ufunc, inputs, domain)):
     ...:         print 'Array:', array
     ...:         print 'Ufunc:', ufunc
     ...:         print 'Inputs:', inputs
     ...:         print 'Domain:', domain
     ...:         return array
     ...:
     ...: m = MyArray((1, 2))

In [160]: np.add([1, 2], [3, 4], out=m)
Array: [[  6.93023165e-310   1.33936849e-316]]
Ufunc: <ufunc 'add'>
Inputs: ([1, 2], [3, 4], MyArray([[  6.93023165e-310,
                                     1.33936849e-316]]))
Domain: 0

Out[160]: MyArray([[ 4.,  6.]])
```

<!-- Internally, buffers are used for misaligned data, swapped data, and data
that has to be converted from one data type to another. The size of internal
buffers is settable on a per-thread basis. There can be up to buffers of the
specified size created to handle the data from all the inputs and outputs of a
ufunc. The default size of a buffer is 10,000 elements. Whenever buffer-based
calculation would be needed, but all input arrays are smaller than the buffer
size, those misbehaved or incorrectly-typed arrays will be copied before the
calculation proceeds. Adjusting the size of the buffer may therefore alter the
speed at which ufunc calculations of various sorts are completed. A simple
interface for setting this variable is accessible using the function

setbufsize(size)        Set the size of the buffer used in ufuncs.
-->
]]

---

.left-column[
  ## Type handling
]

.right-column[

``types`` attribute of ufunc, e.g. ``np.add.types``.

Common types:

```
 * byte -> b
 * short -> h
 * intc -> i
 * double -> d
 * single -> f
 * longfloat -> g
 * complex double -> D
```

See also ``np.sctypeDict``.

If no suitable ufunc loop exists, try to find one to which can be cast safely
(``np.can_cast``).  Can specify casting policy to ufunc via ``casting`` keyword
(``no``, ``equiv``, ``same_kind``, or ``unsafe``--default is the safe 
``same_kind``).  Linear search through available functions--use first match.
]

---

.left-column[
  ## Other ufunc ops
]

.right-column[
.up30[

- ``ufunc.reduce(a[, axis, dtype, out, keepdims])``<br/>
  Reduces a‘s dimension by one, by applying ufunc along one axis.

- ``ufunc.accumulate(array[, axis, dtype, out])``<br/>
  Accumulate the result of applying the operator to all elements.

- ``ufunc.reduceat(a, indices[, axis, dtype, out])``<br/>
  Performs a (local) reduce with specified slices over a single axis.

  ```python
  >>> x = np.arange(12).reshape((3, 4))
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]])

  >>> np.add.reduceat(x, [0, 1, 3], axis=1)
  array([[ 0,  3,  3],
         [ 4, 11,  7],
         [ 8, 19, 11]])
  ```

- ``ufunc.outer(A, B)``<br/>
  Apply the ufunc op to all pairs (a, b) with a in A and b in B.

]]

---

.left-column[
  ## Implementing a ufunc - the kernel
]

.right-column[
.up20[
```
static void square_d(char** args,
                     npy_intp* dimensions,
                     npy_intp* steps,
                     void *data)
{
    npy_intp n = dimensions[0]; /* input length */

    char *input = args[0];
    char *output = args[1];

    npy_intp in_step = steps[0]; /* input stride */
    npy_intp out_step = steps[1]; /* output stride */

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
```

*Credit:* Chris Jordan-Squire, who wrote the numpy user guide entry.
]]

---

.left-column[
  ## Implementing a ufunc - slotting into NumPy
]

.right-column[
``PyUFunc_FromFuncAndData`` - return a universal function based on a
kernel + meta-data
```
// Set up the numpy structures
import_array();
import_umath();

PyUFuncGenericFunction funcs[1] = {&square_d};
void *data[1] = {NULL};
char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

// The above structures are not copied
//    -- must remain visible --

PyObject* ufunc =
  PyUFunc_FromFuncAndData(funcs, data, types,
                          1, /* nr of data types */
                          1, 1, /* nr in & out */
                          PyUFunc_None, /* identity */
                          "square", /* function name */
                          "docstring", /* docstring */
                          0); /* unused */
```
Use the resulting ``ufunc`` inside your module.  See the
[full source](https://github.com/enthought/davidc-scipy-2013/blob/master/examples/ufuncs/my_ufunc.c).
]

---
.left-column[
  ## Implementing a ufunc - provided inner loops
]

.right-column[
.up20[.up30[
```
double kernel(double d) {
    ...;
}

static void *data[] = {&square_d};
static char types[] = {NPY_DOUBLE, NPY_DOUBLE};
static PyUFuncGenericFunction funcs[sizeof(data)];

PyMODINIT_FUNC initmy_ufunc_noloop(void) {
...
    funcs[0] = &PyUFunc_d_d;
    ufunc = PyUFunc_FromFuncAndData(
                    funcs, data, types,
                    1,    /* nr of data types */
                    1, 1, /* nr in & out */
                    PyUFunc_None, /* identity */
                    "square", /* function name */
                    "docstring", /* docstring */
                    0); /* unused */
...
}
```
Examples ([full source](https://github.com/enthought/davidc-scipy-2013/blob/master/examples/ufuncs/my_ufunc_noloop.c)):
```
PyUfunc_f_f(args, dimensions, steps, func)
  float elementwise_func(float in1)
PyUfunc_dd_d
  double elementwise_func(double in1, double in2)

PyUFunc_ff_f_As_dd_d
  double elementwise_func(double in1, double in2)
```
]]]

<!--
Scalar transformation
Combination of arrays -- generalized ufuncs
Reductions
From Cython (Pauli Virtanen):
  http://scipy-lectures.github.io/advanced/advanced_numpy/index.html#universal-functions
-->

<!-- Type resolution -->

---

.left-column[
  ## Implementing a ufunc - multiple types
]

.right-column[.down30[.down30[
You probably need a templating engine.

You probably don't need NumPy's templating engine.

<br/>
Here's [an example using Tempita](https://github.com/enthought/davidc-scipy-2013/blob/master/examples/ufuncs/my_ufunc_types.c.tmpl)
(as well as the
[setup.py](https://github.com/enthought/davidc-scipy-2013/blob/master/examples/ufuncs/setup.py)).

```python
{{
# *NOTE*: Order of this list is important.  First match
#         (even with casting) will be used.
ctypes = ('int8_t', 'int32_t', 'int64_t',
          'float', 'double')
dtypes = ('NPY_INT8', 'NPY_INT32', 'NPY_INT64',
          'NPY_FLOAT', 'NPY_DOUBLE')
}}

{{for type in ctypes}}
static void square_{{type}}(char** args,
             npy_intp* dimensions, npy_intp* steps,
             void *NPY_UNUSED(data))
...
```
]]]

---

.left-column[
  ## Hands-on
]

.right-column[

1. Construct a custom ufunc ``poly(x, a)`` to evaluate the polynomial
  ``(x**2 + 3) - a*x + 5``.
2. Do a timing comparison with a NumPy implementation using IPython's
  ``%timeit``.

]
