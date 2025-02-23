import numpy as np

from app.random_matrix_generator import RandomMatrixGenerator


class HouseholderGenerator(RandomMatrixGenerator):
    def generate_unitary(self, n: int) -> np.ndarray:
        res: np.ndarray = np.identity(n)
        for _ in range(n - 1):
            res = np.dot(res, self.generate_householder(n))
        return np.dot(res, self.generate_phase_matrix(n))

    def generate_householder(self, n: int) -> np.ndarray:
        vec: np.ndarray = np.reshape(self.generate_random_vector(n), (n, 1));
        return np.identity(n) - 2 * np.tensordot(np.transpose(np.conj(vec)), vec, axes=([0], [1]))

    def generate_random_vector(self, n: int) -> np.ndarray:
        # Used conclusions in https://mathworld.wolfram.com/HyperspherePointPicking.html
        vec: np.ndarray = np.random.standard_normal(n) + 1.j * np.random.standard_normal(n)
        return vec / np.linalg.norm(vec)

    def generate_phase_matrix(self, n: int) -> np.ndarray:
        mat: np.ndarray = np.empty((n, n), dtype=np.complex128);
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = np.exp(1.j * np.random.uniform(0, 2 * np.pi, 1))
                else:
                    mat[i, j] = 0
        return mat
