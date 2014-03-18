from numpy cimport npy_intp
from python cimport PyObject

cdef extern from "numpy/arrayobject.h":
    ctypedef struct PyArrayObject

cdef extern from "numpy/ufuncobject.h":

    # MUST ALWAYS BE CALLED FIRST!
    void import_ufunc()

    # defines
    enum:
        UFUNC_ERR_IGNORE
        UFUNC_ERR_WARN
        UFUNC_ERR_RAISE
        UFUNC_ERR_CALL
        UFUNC_ERR_PRINT
        UFUNC_ERR_LOG
    enum:
        UFUNC_MASK_DIVIDEBYZERO = 0x07
        UFUNC_MASK_OVERFLOW     = 0x3f
        UFUNC_MASK_UNDERFLOW    = 0x1ff
        UFUNC_MASK_INVALID      = 0xfff
    enum:
        UFUNC_SHIFT_DIVIDEBYZERO = 0
        UFUNC_SHIFT_OVERFLOW     = 3
        UFUNC_SHIFT_UNDERFLOW    = 6
        UFUNC_SHIFT_INVALID      = 9
    enum:
        UFUNC_FPE_DIVIDEBYZERO  = 1
        UFUNC_FPE_OVERFLOW      = 2
        UFUNC_FPE_UNDERFLOW     = 4
        UFUNC_FPE_INVALID       = 8
    enum:
        UFUNC_ERR_DEFAULT = 0
        UFUNC_MAXIDENTITY = 32
    enum:
        PyUFunc_One  = 1
        PyUFunc_Zero = 0
        PyUFunc_None = -1
    enum:
        UFUNC_REDUCE
        UFUNC_ACCUMULATE
        UFUNC_REDUCEAT
        UFUNC_OUTER

    ctypedef struct PyUFunc_PyFuncData:
        int nin
        int nout
        PyObject *callable

    ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)

    ctypedef struct  PyUFuncObject:
        int nin, nout, nargs
        int identity
        PyUFuncGenericFunction *functions
        void **data
        int ntypes
        int check_return
        char *name, *types
        char *doc
        void *ptr
        PyObject *obj
        PyObject *userloops

    object PyUFunc_FromFuncAndData \
       (PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int)
    int PyUFunc_RegisterLoopForType \
       (PyUFuncObject *, int, PyUFuncGenericFunction, int *, void *)
    int PyUFunc_GenericFunction \
       (PyUFuncObject *, PyObject *, PyObject *, PyArrayObject **)
    void PyUFunc_f_f_As_d_d \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_d_d \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_f_f \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_g_g \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F_As_D_D \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_F_F \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_D_D \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_G_G \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_O_O \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_ff_f_As_dd_d \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_ff_f \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_dd_d \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_gg_g \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_FF_F_As_DD_D \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_DD_D \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_FF_F \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_GG_G \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_OO_O \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_O_O_method \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_OO_O_method \
       (char **, npy_intp *, npy_intp *, void *)
    void PyUFunc_On_Om \
       (char **, npy_intp *, npy_intp *, void *)
    int PyUFunc_GetPyValues \
       (char *, int *, int *, PyObject **)
    int PyUFunc_checkfperr \
       (int, PyObject *, int *)
    void PyUFunc_clearfperr \
       ()
    int PyUFunc_getfperr \
       ()
    int PyUFunc_handlefperr \
       (int, PyObject *, int, int *)
    int PyUFunc_ReplaceLoopBySignature \
       (PyUFuncObject *, PyUFuncGenericFunction, int *, PyUFuncGenericFunction *)
