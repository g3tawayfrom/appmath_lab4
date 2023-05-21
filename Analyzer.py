import time
from functools import wraps

from MatrixDecomposition import MatrixDecomposition
from MatrixGeneration import MatrixGeneration


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running '%s': %s seconds" % (function.__name__, str(t1 - t0)))
        return result

    return function_timer


class Analyzer:

    @staticmethod
    def analyze_tridiagonal():
        Analyzer.analyze_gauss(10, MatrixGeneration.tridiagonal)
        Analyzer.analyze_seidel(10, MatrixGeneration.tridiagonal)
        Analyzer.analyze_gauss(50, MatrixGeneration.tridiagonal)
        Analyzer.analyze_seidel(50, MatrixGeneration.tridiagonal)
        Analyzer.analyze_gauss(100, MatrixGeneration.tridiagonal)
        Analyzer.analyze_seidel(100, MatrixGeneration.tridiagonal)

    @staticmethod
    def analyze_hilbert():
        Analyzer.analyze_gauss(10, MatrixGeneration.hilbert)
        Analyzer.analyze_seidel(10, MatrixGeneration.hilbert)
        Analyzer.analyze_gauss(50, MatrixGeneration.hilbert)
        Analyzer.analyze_seidel(50, MatrixGeneration.hilbert)
        Analyzer.analyze_gauss(100, MatrixGeneration.hilbert)
        Analyzer.analyze_seidel(100, MatrixGeneration.hilbert)

    @staticmethod
    @fn_timer
    def analyze_gauss(n, method):
        print(f'\'{method.__name__}\' {n}')
        matrix = method(n)
        right = MatrixGeneration.right(matrix)
        matrix_decomposition = MatrixDecomposition(matrix)
        gauss = matrix_decomposition.solve_by_gauss(right)

    @staticmethod
    @fn_timer
    def analyze_seidel(n, method):
        print(f'\'{method.__name__}\' {n}')
        matrix = method(n)
        right = MatrixGeneration.right(matrix)
        matrix_decomposition = MatrixDecomposition(matrix)
        seidel = matrix_decomposition.solve_by_seidel(right, 1e-3)
