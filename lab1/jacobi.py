import icontract
import itertools
import numpy as np

def _modified_signum(x: float) -> float:
	return 1 if x >= 0 else -1

@icontract.require(lambda A: A.ndim == 2, 'Matrix has to be 2D.')
@icontract.require(lambda A: A.shape[0] == A.shape[1], 'Matrix hast to be square.')
@icontract.require(lambda A: np.all(np.isclose(A, A.T)), 'Matrix has to be symmetric.')
@icontract.require(lambda A: np.issubdtype(A.dtype, np.number) and not np.issubdtype(A.dtype, np.complexfloating), 'Matrix has to be real.')
@icontract.require(lambda tol: tol > 0, 'Tolerance has to be positive real.')
@icontract.require(lambda max_iterations: isinstance(max_iterations, int), 'Maximum iterations has to be an integer.')
@icontract.require(lambda max_iterations: max_iterations > 0, 'Maximum iterations has to be positive integer.')

@icontract.ensure(lambda A, result: len(result[0]) == A.shape[0], 'Number of eigenvalues has to be equal to size of input matrix.')
@icontract.ensure(lambda A, result: result[1].shape == A.shape, 'Eigenvector matrix shape missmatch.')
@icontract.ensure(lambda max_iterations, result: result[2] <= max_iterations, 'Executed iterations exceeded max_iterations.')
def eig_jacobi(A: np.ndarray, tol: float = 1e-3, max_iterations: int = int(1e3)) -> tuple[np.ndarray, np.ndarray, int, float]:
	"""Calculates eigenvalues and eigenvectors of square symmetrical real valued 2D matrix with Jacobi method.

	Parameters
	----------
	A : np.ndarray
		The symmetrical real valued matrix
	tol : float
		The positive tolerance for error.
	max_iterations : int
		The positive number of maximum iterations for method.

	Returns
	-------
	np.ndarray[float]
		The eigenvalues listed in 1D np.ndarray.
	np.ndarray[float]
		The eigenvectors gathered in 2D np.ndarray.
	int
		The amount of iteration steps to end calculation.
	float
		The error calculated as sum of squared non-diagonal elements of matrix.

	Raises
	------
	icontract.ViolationError
		If one of the conditions defined above function declaration are not met.
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
	
	return np.diag(Ai), Vi, steps, Si