#ifndef _COMMON_H_

#include <Python.h>

#include <quadmath.h>

#define __COMP_NPY_UNUSED __attribute__ ((__unused__))
#define NPY_UNUSED(x) (__NPY_UNUSED_TAGGED ## x) __COMP_NPY_UNUSED

typedef __float128 qdouble;

#endif
