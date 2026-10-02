# Лабораторная работа 1: Метод Якоби для вычисления собственных значений и векторов симметричной вещественной матрицы.

## Содержание
--

 - [Установка](#установка)
 - [Описание](#описание)
 - [Тесты](#Тесты)
 - [Примеры](#Примеры)

## Установка
---

Нужен Python версии >=3.9. Все зависимости указаны в файле `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Описание
---

В ходе проведения данной лабораторной работы, были разработаны
- Python пакет `jacobi_solver` вычисления собственных значений и векторов с помощью метода Якоби;
- Python-скрпит тестов проверки контрактов для пакета вычислений.

Функция `eig_jacobi` из реализованного пакета `jacobi_solver` принимает три аргумента:
1. `A: np.ndarray` - двумерная вещественно-симметричная матрица типа `np.ndarray` из библиотеки `numpy`.
2. `tol: float` - абсолютная погрешность вычислений. Вычисления останавливаются, когда сумма квадратов недиагональных элементов матрицы `A` меньше `tol`.
3. `max_iterations: int` - максимальное число итераций. При достижении его, функция выдаст предупреждение о том, что выставленная погрешность результата не гарантируется.

Функция возвращает объект класса `JacobiResult` с полями:
```python
class JacobiResult(NamedTuple):
	eigenvalues: np.ndarray
	eigenvectors: np.ndarray
	iterations: int
	error: float
```

Смысловое значение полей объекта класса `JacobiResult`:
1. `eigenvalues: np.ndarray` - вектор собственных значений матрицы;
2. `eigenvectors: np.npdarray` - матрица собственных векторов матрицы;
3. `iterations: int` - число итераций метода для достижения полученного результата;
4. `error: float` - ошибка, она же сумма квадратов недиагональных элементов матрицы;

Контракты - ограничения, инварианты, предусловия и постусловия работы части кода. В данной работе использовался модуль `icontract` для написания контрактов функции `eig_jacobi`:
```python
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
def eig_jacobi(...) -> JacobiResult:
	pass
```

Описание контрактов:
1. Предусловие на матрицу `A`: тип `np.ndarray`, 2 измерения, квадратность, симметричность, носитель -- число, причем действительное (не комплексное);
2. Предусловие на ошибку `tol`: действительное число, большее 0;
3. Предусловие на максимальное число итераций `max_iterations`: целое число, большее 0;
4. Постусловие на матрицу `A` и результат `result.eigenvalues: число собственных значений должно быть равно размеру матрицы;
5. Постусловие на матрицу `A` и результат `result.eigenvectors`: размер матрицы собственных значений должен быть равен размеру исходной матрицы;
6. Постусловие на матрицу `max_iterations` и результат `result.iterations`: число итераций не превышает максимально установленное значение;

## Тесты
---

Сами тесты написаны в файле `/lab1/tests/test_jacobi.py`. Команда для запуска тестов:
```bash
cd lab1/
python3 -m pytest -v tests/test_jacobi.py
```

С результатами тестов можно ознакомиться в [TESTS.md](TESTS.md)

## Примеры
---

Небольшой скрипт-пример `/lab1/example.py` для демонстрации работы вычислительного пакета, а также сравнение результатов с уже реализованными вариантами, например `numpy.linalg.eigh`:

```python
# /lab1/example.py
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

def main():
	A = np.array([
	    [0, -1,  2],
	    [-1, 2, -1],
	    [2, -1,  0],
	], dtype=np.float64)

	B = np.array([
	    [0, 0, 0, 1],
	    [0, 1, 1, 0],
	    [0, 1, 1, 0],
	    [1, 0, 0, 0],
	], dtype=np.float64)

	C = np.array([
	    [1, 2,  3, 4, 5],
	    [2, 2,  3, 4, 4],
	    [3, 3, 24, 3, 3],
	    [4, 4,  3, 2, 2],
	    [5, 4,  3, 2, 1],
	])

	matricies = [A, B, C]
	for i, m in enumerate(matricies, 1):
	   print('\n' + '=' * 30 + f' matrix {i} ' + '=' * 30 + '\n')
	   check_eigvalues_difference(m)

if __name__ == '__main__':
	main()
```

Запуск:
```bash
cd lab1/
python3 -m example.py
```

Вывод скрипта:
```
============================== matrix 1 ==============================

Matrix:
[[ 0.0000 -1.0000  2.0000]
 [-1.0000  2.0000 -1.0000]
 [ 2.0000 -1.0000  0.0000]]
Eigenvalues by Jacobi: [-2.0000  0.5858  3.4142]
Eigenvalues by Jacobi: [-2.0000  0.5858  3.4142]
Differece (error): [ 0.0001 -0.0000 -0.0001]
Total absolute error: 0.0002

============================== matrix 2 ==============================

Matrix:
[[0.0000 0.0000 0.0000 1.0000]
 [0.0000 1.0000 1.0000 0.0000]
 [0.0000 1.0000 1.0000 0.0000]
 [1.0000 0.0000 0.0000 0.0000]]
Eigenvalues by Jacobi: [-1.0000  0.0000  1.0000  2.0000]
Eigenvalues by Jacobi: [-1.0000  0.0000  1.0000  2.0000]
Differece (error): [0.0000 0.0000 0.0000 0.0000]
Total absolute error: 0.0000

============================== matrix 3 ==============================

Matrix:
[[ 1  2  3  4  5]
 [ 2  2  3  4  4]
 [ 3  3 24  3  3]
 [ 4  4  3  2  2]
 [ 5  4  3  2  1]]
Eigenvalues by Jacobi: [-5.2361 -0.7639 -0.0000  9.5147 26.4853]
Eigenvalues by Jacobi: [-5.2361 -0.7639 -0.0000  9.5147 26.4853]
Differece (error): [ 0.0000  0.0002 -0.0002 -0.0000 -0.0000]
Total absolute error: 0.0004

```
