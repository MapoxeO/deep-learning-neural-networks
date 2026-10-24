import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from alive_progress import alive_bar

__DEBUG__ = False


if torch.cuda.is_available():
	device_name = 'cuda'
elif torch.backends.mps.is_available():
	device_name = 'mps'
else:
	device_name = 'cpu'

#device_name = 'cpu'
device = torch.device(device_name)



DATA_BASE_PATH = Path('data/')
MODELS_BASE_PATH = Path('models/')
IMGS_BAST_PATH = Path('imgs/')
MODELS_BASE_PATH.mkdir(parents=True, exist_ok=True)
IMGS_BAST_PATH.mkdir(parents=True, exist_ok=True)
DATA_FILE_PATH = DATA_BASE_PATH / 'lipo.csv'



def prepare_data(path: Path) -> tuple:
	df = pd.read_csv(path)
	keys = list(df.keys())
	_, target_value_name, *features = keys
	features_count = len(features)

	Xnp = normalize(df[features].to_numpy())
	Ynp = normalize(df[target_value_name].to_numpy())

	X = torch.tensor(Xnp, dtype=torch.float32).to(device)
	Y = torch.tensor(Ynp, dtype=torch.float32).unsqueeze(1).to(device)

	return df, X, Y, features_count



def make_model(in_features, hidden_features, out_features):
	model = nn.Sequential(
		nn.Linear(in_features, hidden_features),
		nn.ReLU(),
		nn.Linear(hidden_features, out_features),
	)

	return model

def train_model(model, X, Y, learning_rate, epochs, callbacks=..., bar=...):
	model.train()
	criteria = nn.MSELoss().to(device)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	prev_loss = float('inf')

	callback_context = {
		'stop': False,
		'learning_rate': learning_rate,
		'epochs': epochs,
		'bar': bar,
		'epoch': ...,
		'prev_loss': ...,
		'loss': ...,
		'abs_loss_delta': ...,
		'epoch': ...,
	}

	for epoch in range(1, epochs + 1):
		predicted = model(X)
		loss = criteria(predicted, Y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		curr_loss = loss.item()
		abs_loss_delta = abs(curr_loss - prev_loss)

		callback_context['prev_loss'] = prev_loss
		callback_context['loss'] = curr_loss
		callback_context['epoch'] = epoch
		callback_context['abs_loss_delta'] = abs_loss_delta

		if callbacks is not ...:
			for callback in callbacks:
				callback(callback_context)

		if callback_context['stop']:
			break

		if epoch == epochs:
			print(f"Training stopped: reached max epochs: {epochs}\n - epoch: {epoch}\n - loss: {curr_loss}\n - prev_loss: {prev_loss}\n - abs_loss_difference: {abs_loss_delta}")

		prev_loss = curr_loss

def save_model(model, path: Path):
	torch.save(model.state_dict(), path)

def print_model(model, verbose=False):
	for i, (layer_name, weights) in enumerate(model.state_dict().items(), 1):
		print(f'{i}. Layer: {layer_name}, Shape: {weights.shape}')
		if verbose:
			print(weights)

def test_features_impact(model, features_count, X):
	model.eval()
	with torch.no_grad():
		zero_features = torch.zeros((1, features_count), dtype=torch.float32).to(device)
		test_features = torch.mean(X, dim=0, keepdim=True).repeat(features_count, 1)
		for i in range(features_count):
			test_features[i, i] = 1.0
		
		zero_impact = model(zero_features).cpu().numpy().flatten()
		features_impact = model(test_features).cpu().numpy().flatten()

	centered_impact = features_impact - zero_impact
	max_abs = np.max(np.abs(centered_impact))
	max_abs = 1.0 if max_abs == 0 else max_abs
	normalized_impact = centered_impact / max_abs

	return normalized_impact



def normalize(x):
	_min = x.min(axis=0)
	_max = x.max(axis=0)
	_denom = _max - _min
	if x.ndim == 1:
		_denom = 1 if _denom == 0 else _denom
	else:
		_denom[_denom == 0] = 1.0
	x_scaled = (x - _min) / _denom
	return x_scaled

def denormalize(x, original):
	_min = x.min(axis=0)
	_max = x.max(axis=0)
	_denom = _max - _min
	if x.ndim == 1:
		_denom = 1 if _denom == 0 else _denom
	else:
		_denom[_denom == 0] = 1.0

	return _min + _denom * x



def _log_callback(_ctx):
	if _ctx['epoch'] % 100 == 0 and _ctx['bar'] is not ...:
		_ctx['bar'].text = f"Training... Loss: {_ctx['loss']:.6f}"

def _convergence_break(_ctx):
	convergence_delta = 0.01
	if abs(_ctx['loss']) < convergence_delta:
		print(f"Training stopped: reached loss limit: {convergence_delta}\n - epoch: {_ctx['epoch']}\n - loss: {_ctx['loss']}\n - prev_loss: {_ctx['prev_loss']}\n - abs_loss_difference: {_ctx['abs_loss_delta']}")
		_ctx['stop'] = True

def _early_break(_ctx):
	difference_delta = 1e-8
	if _ctx['abs_loss_delta'] < difference_delta:
		print(f"Training stopped: reached absulute loss difference: {difference_delta}\n - epoch: {_ctx['epoch']}\n - loss: {_ctx['loss']}\n - prev_loss: {_ctx['prev_loss']}\n - abs_loss_difference: {_ctx['abs_loss_delta']}")
		_ctx['stop'] = True



def plot(features_name, impacts):
	plt.figure(figsize=(12, 15))

	num_features = len(features_name)
	y_indices = np.arange(num_features)
	total_bar_width = 0.60
	bar_height = total_bar_width / len(impacts)
	colors = ["#ff0000", "#ff8000", "#bebe12", "#00bd06", "#2440bd", "#87005a"]

	for y in y_indices:
		if y < num_features - 1:
			plt.axhline(y = y + 0.5, color="#373737", linewidth=0.75, alpha=0.6)

	for i, impact in enumerate(impacts):
		y_offset = y_indices - (total_bar_width / 2) + (i * bar_height) + (bar_height / 2)
		plt.barh(y_offset, impact, height=bar_height, label=f'Model {i + 1}', 
				color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.85)

	plt.yticks(y_indices, features_name, fontsize=11)
	plt.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
	plt.xlim(-1.1, 1.1)
	
	plt.xlabel('Степень влияния на липофильность', fontsize=12, labelpad=10)
	plt.title(f'Сравнение влияния признаков во всех {len(impacts)} моделях', fontsize=18, pad=15)
	
	plt.grid(axis='x', linestyle='--', alpha=0.4)
	plt.legend(loc='upper left', fontsize=11)

	plt.gca().invert_yaxis()
	plt.tight_layout()

	if __DEBUG__:
		plt.show()
	else:
		plt.savefig(IMGS_BAST_PATH / 'features-impact-plot.pdf')




def main():
	print(f'Currently working on device: {device_name}')

	df, X, Y, features_count = prepare_data(DATA_FILE_PATH)

	get_model_mock = lambda _hidden_features: make_model(features_count, _hidden_features, 1)
	hidden_features = [2, 4, 8, 16, 32, 64]
	models: list[nn.Sequential] = []
	learning_rate = 0.0005
	epochs = 20000

	callbacks = [
		_log_callback,
		_convergence_break,
		_early_break
	]

	print(f'Learning rate: {learning_rate}\nMax epochs: {epochs}\n')

	with alive_bar(len(hidden_features)) as bar:
		for k in hidden_features:
			bar.title = f'Training... Hidden layer neurons: {k}'
			model = get_model_mock(k)
			model.to(device)
			bar.text = 'Training...'
			train_model(model, X, Y, learning_rate, epochs, callbacks=callbacks, bar=bar)
			models.append(model)
			bar()

	features_name = list(df.keys())[2:]
	impacts = []

	for i, model in enumerate(models, 1):
		print('\n' + ''.join(['=' * 25, f' Model {i} ', '=' * 25]))
		print_model(model)
		print()
		impact = test_features_impact(model, features_count, X)
		impacts.append(impact)

	plot(features_name, impacts)


if __name__ == '__main__':
	main()