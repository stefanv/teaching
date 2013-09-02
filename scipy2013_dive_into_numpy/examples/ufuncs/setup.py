from distutils.core import setup
from distutils.extension import Extension

import numpy

def template(fn):
    import tempita

    with open(fn.replace('.tmpl', ''), 'w') as out:
        out.write(tempita.Template.from_filename(fn).substitute())


template('my_ufunc_types.c.tmpl')


extensions = [
    Extension("my_ufunc", ["my_ufunc.c"],
              include_dirs=[numpy.get_include()]),
    Extension("my_ufunc_types", ["my_ufunc_types.c"],
              include_dirs=[numpy.get_include()]),
    Extension("my_ufunc_noloop", ["my_ufunc_noloop.c"],
              include_dirs=[numpy.get_include()])
]


setup(ext_modules=extensions)
