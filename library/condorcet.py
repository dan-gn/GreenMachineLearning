# LIBRARIES
import pandas as pd
import numpy as np

# INPUTS
path = 'results_new_linux_results/'
models = ['ResidualNeuralNetwork', 'MultiLayerNeuralNetwork', 'GradientBoostingClassifier', 'RandomForestClassifier', 'SVC', 'LogisticRegression']
datasets = ['aids', 'malware', 'students']
metrics = ['accuracy', 'auc', 'log_loss', 'energy_train', 'energy_pred', 'total_energy']

# READ RESULTS

for metric in metrics:
	output = np.zeros((6,6))
	for dataset in datasets:
		x = pd.read_csv(path + f'posthoc/{dataset}_{metric}.csv', index_col=0).to_numpy()
		output = output + x
	output = pd.DataFrame(output)
	output.columns = models
	output.index = models
	output['total'] = output.sum()
	output.to_csv(path + f'condorcet/{metric}.csv')

