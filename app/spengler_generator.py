import math

import numpy as np

from app.random_matrix_generator import RandomMatrixGenerator


class SpenglerGenerator(RandomMatrixGenerator):
    def generate_unitary(self, n: int) -> np.ndarray:
        res: np.ndarray = np.identity(n)
        lambdas: np.ndarray = self.generate_lambdas(n)
        for i in range(0, n-1):
            for j in range(i + 1, n):
                res = np.dot(res, np.dot(self.exp_z(i, j, lambdas[j, i], n), self.exp_y(i, j, lambdas[i, j], n)))
        for i in range(0, n-1):
            res = np.dot(res, self.exp_z(i, n, lambdas[i, i], n))
        return res

    def generate_lambdas(self, n: int) -> np.ndarray:
        mat: np.ndarray = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i > j:
                    mat[i, j] = np.random.uniform(0, np.pi, 1)
                elif i < j:
                    mat[i, j] = np.random.uniform(0, np.pi/2, 1)
                else:
                    mat[i, j] = np.random.uniform(0, 2*np.pi, 1)
        return mat

    def exp_z(self, ii: int, jj: int, _lambda: float, n: int) -> np.ndarray:
        mat: np.ndarray = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                if i == j and i == ii:
                    mat[i, j] = math.cos(_lambda) + 1.j * math.sin(_lambda)
                elif i == j and i == jj:
                    mat[i, j] = math.cos(_lambda) - 1.j * math.sin(_lambda)
                elif i == j:
                    mat[i, j] = 1
                else:
                    mat[i, j] = 0
        return mat

    def exp_y(self, ii: int, jj: int, _lambda: float, n: int) -> np.ndarray:
        mat: np.ndarray = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                if i == j and i == ii:
                    mat[i, j] = math.cos(_lambda)
                elif i == j and i == jj:
                    mat[i, j] = math.cos(_lambda)
                elif i == ii and j == jj:
                    mat[i, j] = math.sin(_lambda)
                elif i == jj and j == ii:
                    mat[i, j] = -math.sin(_lambda)
                elif i == j:
                    mat[i, j] = 1
                else:
                    mat[i, j] = 0
        return mat
