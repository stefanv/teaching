import numpy

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('',parent_package,top_path)
    config.add_extension("_quad", ["_quad.c.src",
				   "number.c",
                                   "pyquad.c"],
			include_dirs=[numpy.get_include()],
			libraries=["quadmath"])
    return config

if __name__ == "__main__":
    setup(configuration=configuration)
