import numpy as np
from scipy.sparse import lil_matrix as csr


class MatrixDecomposition:

    def __init__(self, a):
        self.n = len(a)
        self.a = csr(a, dtype=float)
        self.l = csr((self.n, self.n), dtype=float)
        self.u = csr(np.eye(self.n), dtype=float)
        self._create_lu()

    def _create_lu(self):
        for i in range(self.n):
            for j in range(self.n):
                s = sum([self.l[i, k] * self.u[k, j] for k in range(min(j, i))])
                if i <= j:
                    self.u[i, j] = self.a[i, j] - s
                else:
                    self.l[i, j] = (self.a[i, j] - s) / self.u[j, j]

    def solve_by_gauss(self, b):
        # finding solutions ly = b
        y = np.zeros(self.n)
        for i in range(self.n):
            s = sum([self.l[i, j] * y[j] for j in range(i)])
            y[i] = b[i] - s

        # finding solution for ux = y
        x = np.zeros(self.n)
        for i in reversed(range(self.n)):
            s = sum([self.u[i, j] * x[j] for j in range(i + 1, self.n)])
            x[i] = (y[i] - s) / self.u[i, i]

        return x

    def inverse_matrix(self):
        eye = np.eye(self.n)
        inv = np.array([self.solve_by_gauss(eye[i]) for i in range(self.n)])
        return inv.transpose()

    def solve_by_seidel(self, b, eps):
        x = np.zeros(self.n)
        converges = False
        while not converges:
            for i in range(self.n):
                s = sum([self.a[i, j] * x[j] for j in range(self.n) if i != j])
                x[i] = (b[i] - s) / self.a[i, i]
            norm = 0.0
            for i in range(self.n):
                s = sum([self.a[i, j] * x[j] for j in range(self.n)])
                norm += (s - b[i]) ** 2
            if norm <= eps ** 2:
                converges = True

        return x