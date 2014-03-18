from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name='npufuncs',
      version='0.0',
      description='custom ufuncs',
      author='My Name',
      author_email='myname@myname.org',
      license='BSD',
      url='http://mentat.za.net',

      # -----

      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension('lgamma_ufunc', ['lgamma_ufunc.pyx'],
                               include_dirs=[numpy.get_include()])],
)

