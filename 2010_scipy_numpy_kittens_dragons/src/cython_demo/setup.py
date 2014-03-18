from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

module = 'calcmod'

def cext(ext_name):
    return Extension('%s.%s' % (module, ext_name),
                     ['%s/%s.pyx' % (module, ext_name)],
                     include_dirs=[numpy.get_include()])

setup(name=module,
      version='0.0',
      description='Calculator of Epic Proportion',
      author='My Name',
      author_email='myname@myname.org',
      license='BSD',
      url='http://myname.org/calc',

      # -----

      packages=[module],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [cext('calc')]
)

