
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys

N = 100*1000*1000
x = np.linspace(1, 10, N)

def raw():
    """Straight-forward NumPy evaluation of polynomial.

    """
    return (((.25 * x) + .75) * x - 1.5) * x - 2

def inplace(block_size=20000):
    """Blocked evaluation of polynomial.

    """
    y = np.empty(len(x))
    for k in range(len(x) // block_size + 1):
        b, e = k * block_size, (k+1) * block_size
        y[b:e] = x[b:e]
        y[b:e] *= .25
        y[b:e] += .75
        y[b:e] *= x[b:e]
        y[b:e] -= 1.5
        y[b:e] *= x[b:e]
        y[b:e] -= 2

    return y

def bench():
    """Illustrate CPU vs memory trade-off.

    Break up a computation in chunks and benchmark. Small blocks fit into
    cache easily, but the NumPy overhead and the outer Python for-loop takes
    long to execute.  With large blocks, the overhead for NumPy and the
    for-loop is negligible, but the blocks no longer fit into cache, resulting
    in delays.

    Returns
    -------
    block_sizes : list
        Size of the different data chunks.
    times : list
        Execution times.

    """
    times = []
    blocks = np.round(np.logspace(3, 7, num=50))
    for b in blocks:
        times.append(timeit.timeit('cpu_vs_mem.inplace(block_size=%d)' % b,
                                   'import cpu_vs_mem', number=1))
        print('Block size: %d  Execution time: %.2f' % (b, times[-1]))

    return blocks, times


blocks, times = bench()
plt.semilogx(blocks, times, 'o-')
plt.xlabel('Block size [b]')
plt.ylabel('Execution time [s]')
plt.title('CPU vs Memory Benchmark')