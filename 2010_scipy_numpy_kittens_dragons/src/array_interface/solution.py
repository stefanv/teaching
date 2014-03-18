import numpy as np

from mutable_str import MutableString

s = MutableString('abcde')

s.__array_interface__ = {'data' : (s.data_ptr, False), # (ptr, read_only)
                         'shape' : (len(s),),
                         'typestr' : '|u1', # unsigned character
                         }

print 'String before converting to array:', s
sa = np.asarray(s)

print 'String after converting to array:', sa

sa += 2
print 'String after adding "2" to array:', s
