# LIBRARIES
import scikit_posthocs as sp
import pandas as pd
import numpy as np
import statistics

def get_data(filename, metric):
	x = pd.read_csv(filename)
	return x[metric].to_numpy()

# INPUTS
path = 'results_new_linux_results/'
# models = ['ResidualNeuralNetwork', 'MultiLayerNeuralNetwork', 'GradientBoostingClassifier', 'RandomForestClassifier', 'SVC', 'LogisticRegression']
models = ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'GradientBoostingClassifier', 'MultiLayerNeuralNetwork', 'ResidualNeuralNetwork']
datasets = ['aids', 'malware', 'students']
feature_reduction = [False, True]
subsampling = [False, True]
# metrics = ['accuracy', 'auc', 'log_loss', 'energy_train', 'energy_pred', 'total_energy']
metric = 'enery_prep'


# READ RESULTS
results = {}
for dataset in datasets:
	x = {}
	x['dataset'] = dataset
	for model in models:
		x['model'] = model
		for ss in subsampling:
			x['ss'] = ss
			for fr in feature_reduction:
				x['fr'] = fr
				key = f'{dataset}_{ss}_{fr}_{model}'
				filename = path + f'{dataset}_{ss}_{fr}_{model}.csv'
				if metric == 'total_energy':
					energy_train = get_data(filename, 'energy_train')
					energy_pred = get_data(filename, 'energy_pred')
					x['values'] = energy_train + energy_pred
				else:
					x['values'] = get_data(filename, metric) * 1000000
				x['mean'] = np.mean(x['values'])
				x['sigma'] = statistics.stdev(x['values'])
				results[key] = dict(x)

		# # RUN TEST
		# samples = np.array([results[model] for model in models], dtype=np.float32).transpose()
		# p_values = sp.posthoc_miller_friedman(samples)
		# significance = p_values < 0.05

		# # STORE 
		# significance.columns = models
		# significance.index = models
		# output_filename = path + f'posthoc/{dataset}_{metric}.csv'
		# significance.to_csv(output_filename)

		# mean_values = samples.mean(axis=0)
		# mean_values = pd.DataFrame(mean_values)
		# mean_values.columns = ['mean']
		# mean_values.index = models
		# output_filename = path + f'mean/{dataset}_{metric}.csv'
		# mean_values.to_csv(output_filename)

output = []
for dataset in datasets:
	for fr in feature_reduction:
		for ss in subsampling:
			row = f'{dataset} '
			row = row + '& \checkmark ' if ss else row + '& '
			row = row + '& \checkmark ' if fr else row + '& '
			for model in models:
				key = f'{dataset}_{ss}_{fr}_{model}'
				row += f"& {results[key]['mean']:.4f} & $\pm$ {results[key]['sigma']:.2f} "
			row += ' \\\\ '
			output.append(row)				


with open(path + f'tables\\{metric}_table.txt', 'w') as f:
	f.write('\\begin{table*} \n')
	f.write('\\caption{} \n')
	f.write('\\label{tab:} \n')
	f.write('\\centering \n \n')
	f.write('\\begin{tabular}{p{0.08\linewidth} c c r@{\phantom{X}}l r@{\phantom{X}}l r@{\phantom{X}}l r@{\phantom{X}}l r@{\phantom{X}}l r@{\phantom{X}}l} \hline \n')
	f.write('Dataset & Subsampling & PCA & \multicolumn{2}{c}{A} & \multicolumn{2}{c}{B} & \multicolumn{2}{c}{C} & \multicolumn{2}{c}{D} & \multicolumn{2}{c}{E} & \multicolumn{2}{c}{F} \\\\  \hline \n \n')
	for line in output:
		f.write(line)
		f.write('\n')
	f.write('\hline \n \n')
	f.write('\end{tabular} \n')
	f.write('\end{table*} \n')
print(output)
