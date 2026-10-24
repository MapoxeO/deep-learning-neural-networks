import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats

DATA_BASE_PATH = Path('data/')
FILE_PATH = DATA_BASE_PATH / 'lipo.csv'

@dataclass
class LeastSquaresResult:
	parameters: np.ndarray
	residuals:  np.ndarray
	model: callable
	tss: float
	rss: float
	ess: float
	R2: float
	R2_adjusted: float
	s2: float
	s: float
	confidence_intervals: list
	t_stats: np.ndarray
	p_values: np.ndarray
	freedom_degree: int


def linear_least_squares(Xs: np.ndarray, Y: np.ndarray):
	assert np.shape(Xs)[0] == np.shape(Y)[0], 'Number of observation missmatch'

	Y = Y.ravel()
	n = Xs.shape[0]
	k = Xs.shape[1]
	# freedom degrees
	p = n - k - 1
	assert p > 0, 'Overfitted'

	X_with_ones = np.hstack((np.ones((n, 1)), Xs))
	b = np.linalg.pinv(X_with_ones) @ Y

	model = lambda x: b[0] + x @ b[1:]
	Y_predicted = model(Xs)
	Y_mean = np.mean(Y)
	e = Y - Y_predicted
	rss = np.sum(e ** 2)
	tss = np.sum((Y - Y_mean) ** 2)
	ess = np.sum((Y_predicted - Y_mean) ** 2)
	R2 = 1 - rss / tss
	R2_adjusted = 1 - (1 - R2) * (n - 1) / p

	s2 = rss / p
	s = np.sqrt(s2)
	X_inv = np.linalg.pinv(X_with_ones.T @ X_with_ones)
	se_coeffs = s * np.sqrt(np.diag(X_inv))
	t_stats = b / se_coeffs
	p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), p))

	t_crit = stats.t.ppf(0.975, p)
	
	confidence_intervals = []
	
	for i in range(k + 1):
		margin = t_crit * se_coeffs[i]
		lower_bound = b[i] - margin
		upper_bound = b[i] + margin
		confidence_intervals.append((lower_bound, upper_bound))

	return LeastSquaresResult(
		b, e, model, tss, rss, ess, R2, R2_adjusted, s2, s, 
		confidence_intervals, t_stats, p_values, p
	)

def get_significant_features(result: LeastSquaresResult, features: list, alpha: float = 0.05) -> pd.DataFrame:
	"""
	Выделяет статистически значимые признаки, собирает их в таблицу 
	и сортирует по возрастанию p-value (от самых значимых к менее значимым).
	"""
	records = []
	for name, b_val, t_stat, p_val, ci in zip(
		features, result.parameters, result.t_stats, result.p_values, result.confidence_intervals
	):
		# Отбираем только те, у которых p-value < alpha
		if p_val < alpha:
			records.append({
				"Регрессор": name,
				"Оценка b": b_val,
				"t-статистика": t_stat,
				"p-value": p_val,
				"Дов. интервал": f"[{ci[0]:.3f}; {ci[1]:.3f}]"
			})
			
	# Создаем DataFrame для красивого табличного отображения и сортируем по p-value
	df_sig = pd.DataFrame(records)
	if not df_sig.empty:
		df_sig = df_sig.sort_values(by="p-value").reset_index(drop=True)
	return df_sig

def plot_regression_diagnostics(Y_actual: np.ndarray, Y_predicted: np.ndarray, residuals: np.ndarray):
	"""
	Строит диагностические графики для оценки качества линейной регрессии.
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	# 1 график: фактические и предсказанные
	ax1.scatter(Y_actual, Y_predicted, color='royalblue', edgecolors='black', alpha=0.7, label='Данные веществ')
	# Идеальная линия предсказания (диагональ)
	ideal_min = min(Y_actual.min(), Y_predicted.min())
	ideal_max = max(Y_actual.max(), Y_predicted.max())
	ax1.plot([ideal_min, ideal_max], [ideal_min, ideal_max], 'r--', lw=2, label='Идеальное предсказание')
	
	ax1.set_title(r'Фактические vs. Предсказанные значения $\log P$', fontsize=12)
	ax1.set_xlabel(r'Фактический $\log P$', fontsize=10)
	ax1.set_ylabel(r'Предсказанный моделью $\widehat{\log P}$', fontsize=10)
	ax1.grid(linestyle=':', alpha=1)
	ax1.legend()
	
	# 2 график: остатки
	ax2.scatter(Y_predicted, residuals, color='crimson', edgecolors='black', alpha=0.7)
	ax2.axhline(y=0, color='black', linestyle='-', lw=1.5)  # Линия нулевой ошибки

	# 2 * stderr границы 
	std_err = np.std(residuals)
	ax2.axhline(y=2*std_err, color='gray', linestyle=':', alpha=0.5, label=r'+2$\sigma$')
	ax2.axhline(y=-2*std_err, color='gray', linestyle=':', alpha=0.5, label=r'-2$\sigma$')
	
	ax2.set_title(r'График остатков по $\log P$', fontsize=12)
	ax2.set_xlabel(r'Предсказанный $\widehat{\log P}$', fontsize=10)
	ax2.set_ylabel(r'Невязки $e = \log P - \widehat{\log P}$', fontsize=10)
	ax2.grid(linestyle=':', alpha=1)
	ax2.legend()
	
	plt.tight_layout()
	plt.show()

def main():
	df = pd.read_csv(FILE_PATH)
	keys = list(df.keys())
	_, target_value_name, *features = keys
	print("\n--- ЛИНЕЙНАЯ РЕГРЕССИЯ ---")
	print(f"Поиск '{target_value_name}' по {len(features)} параметру(-ам).")
	print(f"Всего примеров для обучения: {len(df)}")
	Xs = df[features].to_numpy()
	Y = df[target_value_name].to_numpy()

	result = linear_least_squares(Xs, Y)
	regs = get_significant_features(result, ['1'] + features)
	
	print("\n--- СТАТИСТИЧЕСКИ ЗНАЧИМЫЕ РЕГРЕССОРЫ ---")
	print(regs)
	
	# Рассчитываем предсказанные значения для построения графиков
	Y_predicted = Y - result.residuals
	
	# Вызов визуализации диагностических графиков
	plot_regression_diagnostics(Y, Y_predicted, result.residuals)

if __name__ == '__main__':
	main()