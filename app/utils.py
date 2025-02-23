import numpy as np


class Utils:
    def generate_random_vector(n: int) -> np.ndarray:
        # Used conclusions in https://mathworld.wolfram.com/HyperspherePointPicking.html
        vec: np.ndarray = np.random.standard_normal(n) + 1.j * np.random.standard_normal(n)
        return vec / np.linalg.norm(vec)