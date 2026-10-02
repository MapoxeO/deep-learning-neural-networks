import pytest
import icontract
import numpy as np
from jacobi_solver import eig_jacobi

@pytest.fixture
def jacobi_solver():
	return eig_jacobi

@pytest.mark.parametrize('case', [
	# DIM
	(np.array([1, 2, 3]), False),
	(np.array([[1], [2], [3]]), False),
	(np.array([[[1]], [[2]], [[3]]]), False),

	# SQUARE
	(np.array([[1, 2, 3], [1, 2, 3]]), False),
	(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), False),

	# SYMMETRIC
	(np.array([[1, 0], [1, 0]]), False),
	(np.array([[1, 0], [0, 1]]), True),

	# REAL
	(np.array([[1+0j, 1+1j], [1+1j, 2+0j]]), False),
	(np.array([[1, 1, 1], [1, 2, 1], [1, 1, 3]]), True),
])
def test_for_matrix_jacobi(jacobi_solver, case):
	matrix, should_pass = case
	if should_pass:
		result = jacobi_solver(matrix)
		assert result is not None, 'jacobi_solver returned empty result'
	else:
		with pytest.raises(icontract.ViolationError):
			jacobi_solver(matrix)

@pytest.mark.parametrize('case', [
	(np.array([[0, -1,  2], [-1, 2, -1], [2, -1,  0]]), np.array([-2.0000, 0.5858, 3.4142]), 1e-5),
	(np.array([[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0]]), np.array([-1, 0, 1, 2]), 1e-5),
	(np.array([[1, 2,  3, 4, 5], [2, 2,  3, 4, 4], [3, 3, 24, 3, 3], [4, 4,  3, 2, 2], [5, 4,  3, 2, 1]]), np.array([-5.2361, -0.7639, -0.0000, 9.5147, 26.4853]), 1e-4)
])
def test_for_rightness_jacobi(jacobi_solver, case):
	matrix, to_be, tol = case
	result = jacobi_solver(matrix)
	assert np.all(np.isclose(np.sort(result.eigenvalues), np.sort(to_be), atol=tol)), 'Result was not close to to_be'

@pytest.mark.parametrize('case',
[
	(np.array([[1, 0], [0, 1]]), None, False),
	(np.array([[1, 0], [0, 1]]), 1+3j, False),
	(np.array([[1, 0], [0, 1]]), -1.0, False),
	(np.array([[1, 0], [0, 1]]), 0, False),
	(np.array([[1, 0], [0, 1]]), 1e-6, True),
])
def test_for_tolerance_jacobi(jacobi_solver, case):
	matrix, tol, should_pass = case
	if should_pass:
		result = jacobi_solver(matrix, tol=tol)
		assert result is not None, 'jacobi_solver returned empty result'
	else:
		with pytest.raises(icontract.ViolationError):
			jacobi_solver(matrix, tol=tol)

@pytest.mark.parametrize('case',
[
	(np.array([[1, 0], [0, 1]]), None, False),
	(np.array([[1, 0], [0, 1]]), 1+3j, False),
	(np.array([[1, 0], [0, 1]]), -1.0, False),
	(np.array([[1, 0], [0, 1]]), 0.0, False),
	(np.array([[1, 0], [0, 1]]), 10, True),
])
def test_for_maxit_jacobi(jacobi_solver, case):
	matrix, maxit, should_pass = case
	if should_pass:
		result = jacobi_solver(matrix, max_iterations=maxit)
		assert result is not None, 'jacobi_solver returned empty result'
	else:
		with pytest.raises(icontract.ViolationError):
			jacobi_solver(matrix, max_iterations=maxit)