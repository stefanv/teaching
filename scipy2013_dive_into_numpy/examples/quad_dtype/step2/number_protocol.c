#include "common.h"

static inline qdouble
quad_add(qdouble x, qdouble y)
{
    return x + y;
}

#define _QUAD_BINOP_IMPL(name, binop) \
    PyObject* \
    pyquad_##name(PyObject* a, PyObject* b) { \
        qdouble x, y; \
        x = PyQuad_AsQuad(a); \
        y = PyQuad_AsQuad(b); \
        return PyQuad_FromQuad(binop(x, y)); \
    }

#define quad_BINOP(name) _QUAD_BINOP_IMPL(name, quad_##name)

quad_BINOP(add)

PyNumberMethods pyquad_as_number = {
    pyquad_add,          /* nb_add */
    0, /* nb_subtract */
    0, /* nb_multiply */
    0, /* nb_divide */
    0, /* nb_remainder */
    0, /* nb_divmod */
    0, /* nb_power */
    0, /* nb_negative */
    0, /* nb_positive */
    0, /* nb_absolute */
    0, /* nb_nonzero */
    0, /* nb_invert */
    0, /* nb_lshift */
    0, /* nb_rshift */
    0, /* nb_and */
    0, /* nb_xor */
    0, /* nb_or */
    0, /* nb_coerce */
    0, /* nb_int */
    0, /* nb_long */
    0, /* nb_float */
    0, /* nb_oct */
    0, /* nb_hex */
    0, /* nb_inplace_add */
    0, /* nb_inplace_subtract */
    0, /* nb_inplace_multiply */
    0, /* nb_inplace_divide */
    0, /* nb_inplace_remainder */
    0, /* nb_inplace_power */
    0, /* nb_inplace_lshift */
    0, /* nb_inplace_rshift */
    0, /* nb_inplace_and */
    0, /* nb_inplace_xor */
    0, /* nb_inplace_or */
    0, /* nb_floor_divide */
    0, /* nb_true_divide */
    0, /* nb_inplace_floor_divide */
    0, /* nb_inplace_true_divide */
    0, /* nb_index */
};
