from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

extensions = [
    Extension("poly", ["poly.c"],
              include_dirs=[numpy.get_include()]),
    Extension("poly_cy", ["poly_cy.pyx"])
]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=extensions
)
