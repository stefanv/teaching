# See http://thread.gmane.org/gmane.comp.python.numeric.general/8459

import numpy as np

data = np.random.randn(2000, 39)

def kmean0(data):
    nclusters = 64
    code = data[0:nclusters,:]
    return np.sum((data[..., np.newaxis, ...] - code)**2, 2).argmin(axis=1)

def kmean1(data):
    nclusters = 64
    code = data[0:nclusters,:]
    z = data[:,np.newaxis,:]
    z = z - code
    z = z**2
    z = np.sum(z, 2)
    return z.argmin(axis=1)

def kmean2(data):
    nclusters = 64
    naxes = data.shape[-1]
    code = data[0:nclusters,:]
    data = data[:, np.newaxis]
    allz = np.zeros([len(data)])
    for i, x in enumerate(data):
        z = (x - code)
        z **= 2
        allz[i] = z.sum(-1).argmin(0)
    return allz

def kmean3(data):
    nclusters = 64
    naxes = data.shape[-1]
    code = data[0:nclusters]
    totals = np.zeros([nclusters, len(data)], float)
    transdata = data.transpose().copy()
    for cluster, tot in zip(code, totals):
        for di, ci in zip(transdata, cluster):
            delta = di - ci
            delta **=2
            tot += delta
    return totals.argmin(axis=0)

def kmean4(data):
    nclusters = 64
    naxes = data.shape[-1]
    code = data[:nclusters]
    transdata = data.transpose().copy()
    totals = np.empty([nclusters, len(data)], float)
    code2 = (code**2).sum(-1)
    code2 *= -0.5
    totals[:] = code2[:, np.newaxis]
    for cluster, tot in zip(code, totals):
        for di, ci in zip(transdata, cluster):
            tot += di*ci
    return totals.argmax(axis=0)

if __name__ == '__main__':
    assert np.alltrue(kmean0(data) == kmean1(data))
    assert np.alltrue(kmean0(data) == kmean2(data))
    assert np.alltrue(kmean0(data) == kmean3(data))
    assert np.alltrue(kmean0(data) == kmean4(data))

    from timeit import Timer
    print Timer('kmean0(data)', 'from __main__ import kmean0, data').timeit(3)
    print Timer('kmean1(data)', 'from __main__ import kmean1, data').timeit(3)
    print Timer('kmean2(data)', 'from __main__ import kmean2, data').timeit(3)
    print Timer('kmean3(data)', 'from __main__ import kmean3, data').timeit(3)
    print Timer('kmean4(data)', 'from __main__ import kmean4, data').timeit(3)

