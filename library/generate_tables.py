# Libraries
import numpy as np
import pandas as pd

def get_experiment_name(ss_opt, fr_opt):
	if ss_opt == 'True' and fr_opt == 'True':
		return 'SS & FR'
	elif ss_opt == 'True':
		return 'SS'
	elif fr_opt == 'True':
		return 'FR'
	else:
		return 'Full'
	
# Datasets
datasets = ['aids', 'students', 'malware']

# Machine Learning models
models = ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'GradientBoostingClassifier','ResidualNeuralNetwork', 'MultiLayerNeuralNetwork']

# Experiment options
subsampling_options = [False, True]
feature_reduction_options = [False, True]

# Metric
metric = 'accuracy'

results = []
for dataset in datasets:
	for model in models:
		for ss_opt in subsampling_options:
			for fr_opt in feature_reduction_options:
				filename = f'{dataset}_{ss_opt}_{fr_opt}_{model}'
				path = f'results/{filename}.csv'
				df = pd.read_csv(path)
				res = {}
				res['dataset'] = dataset
				res['ss_opt'] = ss_opt
				res['fr_opt'] = fr_opt
				res['model'] = model
				res['mean'] = df[metric].mean()
				res['std'] = df[metric].std()

				results.append(res)

# Let's get just the results of the full datasets
results = [x for x in results if not(x['ss_opt'] or x['fr_opt'])]

table = {dataset:{model:None for model in models} for dataset in datasets}

for x in results:
	table[x['dataset']][x['model']] = x['mean']

df = pd.DataFrame.from_dict(table).transpose()
print(df)

df.to_csv('final_results/mean_results.csv')