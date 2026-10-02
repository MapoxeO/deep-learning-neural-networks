from typing import NamedTuple
import icontract
import itertools
import numpy as np

class JacobiResult(NamedTuple):
	"""Result object for eig_jacobi function."""
	eigenvalues: np.ndarray    # 1D-array (n,) of eigenvalues
	eigenvectors: np.ndarray   # 2D-array (n, n) of eigenvectors
	iterations: int            # Iteration steps count
	error: float               # The error calculated as sum of squared non-diagonal elements

def _modified_signum(x: float) -> float:
	return 1 if x >= 0 else -1

def _is_valid_matrix(A) -> bool:
	return (isinstance(A, np.ndarray)) == ( A.ndim == 2 ) and ( len(set(A.shape)) == 1 ) and ( np.all(np.isclose(A, A.T)) ) and ( np.issubdtype(A.dtype, np.number) and not np.issubdtype(A.dtype, np.complexfloating) )

def _is_valid_tol(tol) -> bool:
	return isinstance(tol, float) and tol > 0

def _is_valid_maxit(max_iterations) -> bool:
	return isinstance(max_iterations, int) and max_iterations > 0

@icontract.require(_is_valid_matrix, 'Matrix A has to be 2D real symmetric.')
@icontract.require(_is_valid_tol, 'Tolerance has to be positive real.')
@icontract.require(_is_valid_maxit, 'Maximum iterations has to be a positive integer.')

@icontract.ensure(lambda A, result: len(result.eigenvalues) == A.shape[0], 'Number of eigenvalues has to be equal to size of input matrix.')
@icontract.ensure(lambda A, result: result.eigenvectors.shape == A.shape, 'Eigenvector matrix shape missmatch.')
@icontract.ensure(lambda max_iterations, result: result.iterations <= max_iterations, 'Executed iterations exceeded max_iterations.')
def eig_jacobi(A: np.ndarray, tol: float = 1e-3, max_iterations: int = 1000) -> JacobiResult:
	"""Calculates eigenvalues and eigenvectors of square symmetrical real valued 2D matrix with Jacobi method.

	Parameters
	----------
	A : np.ndarray
		The symmetrical real valued matrix.
	tol : float
		The positive tolerance for error.
	max_iterations : int
		The positive number of maximum iterations for method.

	Returns
	-------
	JacobiResult
		A named tuple containing:
		- eigenvalues : (n,) ndarray of eigenvalues.
		- eigenvectors : (n, n) ndarray where each column is an eigenvector.
		- iterations : int, the number of iteration steps executed.
		- error : float, the final error calculated as sum of squared non-diagonal elements.

	Raises
	------
	icontract.ViolationError
		If any of the preconditions (matrix symmetry, dimensions, types) 
		or postconditions (invariants, output shapes) are violated.
	"""

	Ai = A.astype(float).copy() 
	n = Ai.shape[0]
	Vi = np.eye(n)
	
	non_diagonal = ~np.eye(n, dtype=bool)
	upper_diagonal_indecies = list(zip(*np.triu_indices(n, k=1)))
	num_upper_diagonal_indecies = len(upper_diagonal_indecies)

	skipped = 0
	Si = np.sum(Ai[non_diagonal] ** 2)
	steps = 0

	for p, q in itertools.cycle(upper_diagonal_indecies):
		if Si < tol:
			# convergence reached
			break
		if steps >= max_iterations:
			print('Warning! Exceeded maximum amount of iterations. Convergence may not be reached!')
			break
		if skipped >= num_upper_diagonal_indecies:
			print('Warning! Skipped all upper diagonal elements: all elements are bellow the threshold.')
			break
		if Ai[p, q] ** 2 < tol / (2 * num_upper_diagonal_indecies):
			# skipping zero-like elements
			skipped += 1
			continue

		skipped = 0
		removing = Ai[p, q]

		Ci = (Ai[q, q] - Ai[p, p]) / (2 * removing)
		tan = 1 / (Ci + _modified_signum(Ci) * np.sqrt(Ci ** 2 + 1))			
		cos = 1 / np.sqrt(1 + tan ** 2)
		sin = tan * cos

		Ai[p, q] = Ai[q, p] = 0
		Ai[p, p] -= removing * tan
		Ai[q, q] += removing * tan
		
		for r in range(n):
			if r == p or r == q:
				continue
			a_rp = Ai[r, p]
			a_rq = Ai[r, q]
			
			# rotate specific matrix elements
			Ai[r, p] = Ai[p, r] = cos * a_rp - sin * a_rq
			Ai[r, q] = Ai[q, r] = sin * a_rp + cos * a_rq

		v_p = Vi[:, p].copy()
		v_q = Vi[:, q].copy()
		Vi[:, p] = cos * v_p - sin * v_q
		Vi[:, q] = sin * v_p + cos * v_q

		steps += 1
		Si -= 2 * (removing) ** 2
	
	result = JacobiResult(np.diag(Ai), Vi, steps, Si)
	return result