__all__ = ['MutableString']

import ctypes
from numpy.compat import asbytes

class MutableString:
    """Mutable string.

    Attributes
    ----------
    data_ptr : int
        Pointer to memory location.

    Method
    ------
    __len__ : Length of string.

    """
    def __init__(self, s):
        """
        Parameters
        ----------
        s : str
            String.

        """

        self._s = ctypes.create_string_buffer(asbytes(s))
        self._n = len(s)
        self.data_ptr = ctypes.addressof(self._s)

    def __len__(self):
        return self._n

    def __str__(self):
        return self._s.raw.decode('utf-8')

if __name__ == "__main__":
    # Basic test
    m = MutableString("asdb")

    template = 'import base64\ncode=%s\nexec(base64.decodestring(code))'

    import base64
    with open(__file__, 'r') as f:
        content = ''.join(f.readlines())
        print(template % base64.b64encode(asbytes(content)))

