import numpy as np

#@profile
def forloop():
    out = np.empty((1024, 1024))
    for i in range(1024):
        for j in range(1024):
            out[i, j] = i + j
    return out

#@profile
#def forloop():
#    x, y = np.ogrid[:1024, :1024]
#    return x + y

print "Welcome to matrix manipulator v1.0!"
for n in range(5):
    forloop()

print "Computation completed."

#assert np.all(forloop() == numpyloop())

