import sys

import numpy as np
import matplotlib.pyplot as plt

from app.householder_generator import HouseholderGenerator
from app.givens_generator import GivensGenerator
from app.spengler_generator import SpenglerGenerator
from app.random_matrix_generator import RandomMatrixGenerator


def main():
    unitaries: np.ndarray = None;
    class_ = getattr(sys.modules[__name__], sys.argv[1])
    n: int = int(sys.argv[2])
    no_iters: int = int(sys.argv[3])
    out_path: str = sys.argv[4]
    generator: RandomMatrixGenerator = class_()
    for i in range(no_iters):
        unitary: np.ndarray = np.reshape(generator.generate_unitary(n), (n, n, 1))
        if unitaries is None:
            unitaries = unitary
        else:
            unitaries = np.concatenate((unitaries, unitary), axis=2)
    plot_histogram(unitaries, n, out_path)

def plot_histogram(unitaries: np.ndarray, n: int, out_path: str):
    figure, axis = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            axis[i, j].scatter(unitaries[i, j].real, unitaries[i, j].imag, s=1)
    plt.savefig(out_path)

def display_matrix(matrix: np.ndarray):
    rows, columns = matrix.shape

    w = 10
    h = 10
    plt.figure(1, figsize=(w, h))
    tb = plt.table(cellText=np.round(matrix, 2), loc=(0, 0), cellLoc='center')

    tc = tb.properties()['celld']
    for cell in tc.values():
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def display_vector(vector: np.ndarray):
    columns = len(vector)

    w = 10
    h = 1
    plt.figure(1, figsize=(w, h))
    tb = plt.table(cellText=[np.round(vector, 2)], loc=(0, 0), cellLoc='center')

    tc = tb.properties()['celld']
    for cell in tc.values():
        cell.set_width(1.0 / columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

if __name__ == "__main__":
    main()
