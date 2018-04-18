import numpy as np


def f(x):
    return x**4 - 3*x


def integrate_f(a, b, N):
    x = np.linspace(a, b, N)
    dx = (b - a) / N

    return np.sum(f(x)) * dx


if __name__ == "__main__":
    print(integrate_f(0, 1, 10000))
