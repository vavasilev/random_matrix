import numpy as np

from app.random_matrix_generator import RandomMatrixGenerator
from app.utils import Utils


class GivensGenerator(RandomMatrixGenerator):
    def generate_unitary(self, n: int) -> np.ndarray:
        res: np.ndarray = np.identity(n)
        for i in range(0, n-1):
            for j in range(i + 1, n):
                res = np.dot(res, self.generate_givens(i, j, n))
        return res

    def generate_givens(self, ii: int, jj: int, n: int) -> np.ndarray:
        vec: np.ndarray = Utils.generate_random_vector(2)

        mat: np.ndarray = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                if i == ii and j == ii:
                    mat[i, j] = vec[0]
                elif i == ii and j == jj:
                    mat[i, j] = vec[1]
                elif i == jj and j == ii:
                    mat[i, j] = -np.conj(vec[1])
                elif i == jj and j == jj:
                    mat[i, j] = np.conj(vec[0])
                elif i == j:
                    mat[i, j] = 1
                else:
                    mat[i, j] = 0
        return mat

