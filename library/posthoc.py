# LIBRARIES
import scikit_posthocs as sp
import pandas as pd
import numpy as np

def get_data(filename, metric):
	x = pd.read_csv(filename)
	return x[metric].to_numpy()

# INPUTS
path = 'results_new_linux_results/'
models = ['ResidualNeuralNetwork', 'MultiLayerNeuralNetwork', 'GradientBoostingClassifier', 'RandomForestClassifier', 'SVC', 'LogisticRegression']
datasets = ['aids', 'malware', 'students']
metrics = ['accuracy', 'auc', 'log_loss', 'energy_train', 'energy_pred', 'total_energy']

# READ RESULTS
for dataset in datasets:
	for metric in metrics:
		results = {}
		for model in models:
			filename = path + f'{dataset}_False_False_{model}.csv'
			if metric == 'total_energy':
				energy_train = get_data(filename, 'energy_train')
				energy_pred = get_data(filename, 'energy_pred')
				results[model] = energy_train + energy_pred
			else:
				results[model] = get_data(filename, metric)

		# RUN TEST
		samples = np.array([results[model] for model in models], dtype=np.float32).transpose()
		p_values = sp.posthoc_miller_friedman(samples)
		significance = p_values < 0.05

		# STORE 
		significance.columns = models
		significance.index = models
		output_filename = path + f'posthoc/{dataset}_{metric}.csv'
		significance.to_csv(output_filename)

		mean_values = samples.mean(axis=0)
		mean_values = pd.DataFrame(mean_values)
		mean_values.columns = ['mean']
		mean_values.index = models
		output_filename = path + f'mean/{dataset}_{metric}.csv'
		mean_values.to_csv(output_filename)

