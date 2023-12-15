# Libraries
import numpy as np
import os
import sys
import csv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn

# Get path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Datasets
datasets = ['aids', 'students', 'malware']

# Machine Learning models
models = ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'GradientBoostingClassifier','DeepNeuralNetwork']

# Experiment options
subsampling_options = ['False']
feature_reduction_options = ['False']
N_EXPERIMENTS = 40

# Inputs
DATASET = datasets[2]
METRIC = 'accuracy'

results = {}
for model in models:
    for ss_opt in subsampling_options:
        for fr_opt in feature_reduction_options:
            filename = f'{DATASET}_{ss_opt}_{fr_opt}_{model}'
            path = f'{parent}/results/{filename}.csv'
            df = pd.read_csv(path)
            results[filename] = df[METRIC].transpose()

x = []
for key in results:
    x.extend([key for _ in range(N_EXPERIMENTS)])

fig = go.Figure()
for model in models:
    for ss_opt in subsampling_options:
        for fr_opt in feature_reduction_options:
            filename = f'{DATASET}_{ss_opt}_{fr_opt}_{model}'
            y = []
            y.extend(list(results[filename]))
            fig.add_trace(go.Box(y=y, x=x, name=filename))

fig.update_layout(
    xaxis_title='Experiment',
    yaxis_title=METRIC,
    boxmode='group'
)
fig.show()
