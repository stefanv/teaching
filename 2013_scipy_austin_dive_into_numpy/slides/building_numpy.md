---
template: inverse

# 2. Building NumPy

---
layout:false

.left-column[
  ## Building with distutils
]

.right-column[

Simple in-place build with default compiler

```bash
$ python setup.py build_ext -i
```

Running the test suite on numpy.core

```bash
$ nosetests numpy/core
```
]

---
template:inverse

# That was not too hard !

<!-- another pic here -->

---
template:inverse
# Building with Bento

---

layout:false
.left-column[
 ## Building with Bento
 ### Installation and configuration
]

.right-column[

Setting up Bento requires a few steps:

(It is recommended to do this in a virtualenv)

```bash
git clone https://github.com/cournape/Bento \
	~/src/bento-git

cd ~/src/bento-git
python setup.py install --user

git clone https://code.google.com/p/waf ~/src/waf-git
# Tells bento where to look for waf
# (waf has no setup.py)
export WAFDIR=~/src/waf-git

# In NumPy source tree, do an in-place build
bentomaker build -i
```

Set up the Python path (only done once):

```bash
export PYTHONPATH=$PYTHONPATH:~/src/numpy-git
```

]

---

.left-column[
  ## Building with Bento
  ### Features for development
]

.right-column[

Bento is nifty for NumPy development.

- Parallel builds
```bash
bentomaker build -i -j4
```

- Reliable partial rebuilds:
```bash
bentomaker build -i -j4
# Hack to bypass autoconfigure
bentomaker --disable-autoconfigure build -i -j4
```

- Easy compilation customization:
```bash
CC=clang CFLAGS="-O0 -g -Wall -Wextra" \
	bentomaker build -i -j4
```
]

---
template:inverse
# Practice
---

layout:false
.left-column[
 ## A first exercise
]

.right-column[

After setting up Bento, build NumPy with warning on

```
CFLAGS="-O0 -g -Wall -Wextra -W" bentomaker build -i
```

Lots of warnings of the type:

```
../numpy/linalg/lapack_litemodule.c:863:58: warning: \
	unused parameter 'args' [-Wunused-parameter]
lapack_lite_xerbla(PyObject *NPY_UNUSED(self), \
	PyObject *args)
1 warning generated.
```

NumPy has a special macro to decorate unused argument and give an error if they are used

```
void foo(PyObject *NPY_UNUSED(self), ...)
```

Try fixing one .c file so that there is no warning anymore
]
