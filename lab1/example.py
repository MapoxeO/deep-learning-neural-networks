from jacobi_solver import *
import numpy as np

np.set_printoptions(floatmode='fixed', precision=4, suppress=True)

def check_eigvalues_difference(A: np.ndarray):
    result = eig_jacobi(A, tol=1e-2)
    eig_jacobian = np.sort(result.eigenvalues)
    eig_numpy = np.sort(np.linalg.eigh(A).eigenvalues)
    error = eig_jacobian - eig_numpy
    summed_abs_error = np.sum(np.abs(error))

    print(f'Matrix:\n{A}')
    print(f'Eigenvalues by Jacobi: {eig_numpy}')
    print(f'Eigenvalues by Jacobi: {eig_numpy}')
    print(f'Differece (error): {error}')
    print(f'Total absolute error: {summed_abs_error:0.4f}')

A = np.array([
    [0, -1,  2],
    [-1, 2, -1],
    [2, -1,  0],
], dtype=np.float64)

B = np.array([
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
], dtype=np.float64)

C = np.array([
    [1, 2,  3, 4, 5],
    [2, 2,  3, 4, 4],
    [3, 3, 24, 3, 3],
    [4, 4,  3, 2, 2],
    [5, 4,  3, 2, 1],
])

matricies = [A, B, C]

for i, m in enumerate(matricies, 1):
    print('\n' + '=' * 30 + f' matrix {i} ' + '=' * 30 + '\n')
    check_eigvalues_difference(m)