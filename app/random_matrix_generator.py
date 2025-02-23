from abc import ABC, abstractmethod

import numpy as np


class RandomMatrixGenerator(ABC):
    @abstractmethod
    def generate_unitary(self, n: int) -> np.ndarray:
        pass