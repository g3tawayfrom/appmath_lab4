from random import randrange

import numpy as np
from scipy.sparse import diags


class MatrixGeneration:

    @staticmethod
    def tridiagonal(n):
        r = randrange(2, 10)
        k = [(r // 2) * np.ones(n - 1), r * np.ones(n), (r // 2) * np.ones(n - 1)]
        offset = [-1, 0, 1]
        return diags(k, offset).toarray()

    @staticmethod
    def hilbert(n):
        return np.array([[1 / (i + j + 1) for i in range(n)] for j in range(n)])

    @staticmethod
    def right(a):
        f = np.array([sum(a[i, j] * (j + 1) for j in range(len(a))) for i in range(len(a))])
        return f