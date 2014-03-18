from calcmod import calc
import numpy as np

from numpy.testing import *

def test_compute_arr():
    x = np.zeros((5, 5), dtype=int)
    calc.compute_arr(x)

    a, b = np.ogrid[:5, :5]
    assert_array_equal(x, a + b)


