import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
from collections.abc import Callable
from pathlib import Path
from alive_progress import alive_bar

__DEBUG__ = False


IMGS_PATH = Path('imgs/')
IMGS_PATH.mkdir(parents=True, exist_ok=True)


def _normal_dist(shape: tuple[int, int], mu: float, sigma: float) -> np.ndarray:
	return np.random.normal(size=shape, loc=mu, scale=sigma)

def _bernoulli_dist(shape: tuple[int, int], X: np.ndarray, W: np.ndarray = None) -> np.ndarray:
	if not isinstance(X, np.ndarray):
		X = np.array(X)

	if not W:
		return np.random.choice(X, shape)
	return np.random.choice(X, shape, W)

def _vigner_dist_pdf(s: np.ndarray):
	return (np.pi * s / 2) * np.exp(- np.pi * s ** 2 / 4)



class Distribution(NamedTuple):
	type: str
	title: str
	gen: Callable[[tuple[int, int]], np.ndarray]

def generate_ensemble(n: int, k: int, dist: Distribution) -> list[np.ndarray]:
	result = [0] * n
	for i in range(n):
		matrix: np.ndarray = dist.gen((k, k)).astype(np.float64)
		result[i] = (matrix + matrix.T) / 2
	
	return result

def get_normalized_eigenvalues_differnces(ensemble: list[np.ndarray]) -> np.ndarray:
	result = [0] * len(ensemble)
	for i in range(len(ensemble)):
		eigv = np.sort(np.linalg.eigh(ensemble[i]).eigenvalues)
		eigv_diffs = eigv[1] - eigv[0]
		result[i] = eigv_diffs

	result = np.array(result).flatten()
	result /= np.mean(result)

	return result

def make_graph(data: np.ndarray, n: int, k: int, dist: Distribution):
	X = np.linspace(0, 6, 250)
	Y = _vigner_dist_pdf(X)
	#bins = 100
	bins = 1 + int(10 * (len(set(data)) / 10 ) ** 0.25)

	plt.close()
	plt.figure(figsize=(8, 6))
	plt.title('\n'.join([r'Распределение разности $\Delta = \lambda_{2} - \lambda_{1}$ собственных значений', rf'${n}$ матриц размера ${k}\times{k}$.', dist.title]))
	plt.hist(data, bins=bins, density=True, label='Сгенерированные данные')
	plt.plot(X, Y, label='Распределение Вигнера:\n' + r'$\rho_\Delta (s) = \frac{\pi s}{2} e^{-\frac{\pi s^2}{4}}$')
	plt.xlabel(r'Значение $s$ случайной величны $\Delta$')
	plt.ylabel(r'Плотность распределения $\rho_\Delta$')
	plt.legend()
	plt.grid()
	plt.tight_layout()
	if __DEBUG__:
		plt.show()
	else:
		plt.savefig(IMGS_PATH / f'{dist.type}-{n}-{k}.pdf')



def main():
	distributions = [
		Distribution('normal', 'Элементы матриц распределены нормально:\n' + r'$a_{ij} \sim \mathcal{N}(\mu = 0, \sigma = 1)$', gen=lambda x: _normal_dist(x, 0, 1)),
		Distribution('bernoulli', 'Элементы матриц распределены по Бернулли:\n' + r'$\rho_{a_{ij}}(x) = ^1\!/\!_2 (\delta(x-1) + \delta(x+1))$', gen=lambda x: _bernoulli_dist(x, [-1, 1])),
	]
	N = 10000
	studies = [
		(N, 2, distributions[0]),
		(N, 4, distributions[0]),
		(N, 16, distributions[0]),
		(N, 32, distributions[0]),
		(N, 64, distributions[0]),

		(N, 2, distributions[1]),
		(N, 4, distributions[1]),
		(N, 16, distributions[1]),
		(N, 32, distributions[1]),
		(N, 64, distributions[1]),
	]

	with alive_bar(len(studies), theme='classic') as bar:
		for study in studies:
			n, k, dist = study
			bar.text = 'Generating the ensemble...'
			ensemble = generate_ensemble(n, k, dist)
			bar.text = 'Calculating eigenvalues of the ensemble...'
			normalized_diffs = get_normalized_eigenvalues_differnces(ensemble)
			bar.text = 'Plotting distribution...'
			make_graph(normalized_diffs, n, k, dist)
			bar()

if __name__ == '__main__':
	main()