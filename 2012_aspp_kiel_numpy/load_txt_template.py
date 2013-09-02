import numpy as np

txtdata = open('data.txt', 'r')

# Construct the data-type
# In IPython type np.dtype?<ENTER> for examples, e.g.,
#
# np.dtype([('x', np.float), ('y', np.int), ('z', np.uint8)])

dt = np.dtype(...) # Modify this line to give the correct answer

# Load data with loadtxt
data = np.loadtxt(txtdata, dtype=dt)

print data
