# Utilities libraries
import numpy as np
import types

# Machine learning pipeline libraries
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Metrics libraries
import time
from sklearn.metrics import accuracy_score, log_loss
from sklearn import metrics 

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier        
from kdnn import KDNN_v5

# Code Carbon
from codecarbon import OfflineEmissionsTracker, EmissionsTracker

# Utilities
def is_instance_attr(obj, name):
   if not hasattr(obj, name):
      return False
   if name.startswith("__") and name.endswith("__"):
      return False
   v = getattr(obj, name)
   if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
      return False
   # See https://stackoverflow.com/a/17735709/
   attr_type = getattr(type(obj), name, None)
   if isinstance(attr_type, property):
      return False
   return True

def get_instance_attrs(obj):
   names = dir(obj)
   names = [name for name in names if is_instance_attr(obj, name)]
   return names

def sklearn_sizeof(obj):
   sum = 0
   names = get_instance_attrs(obj)
   for name in names:
      v = getattr(obj, name)
      v_type = type(v)
      v_sizeof = v.__sizeof__()
      sum += v_sizeof
   return sum

# Constants
N_FOLDS = 10
N_REPEATS = 2 # CHANGE THIS TO 3 FOR FINAL EXPERIMENTS
TRACKER = OfflineEmissionsTracker(project_name="green_ML", country_iso_code="IRL", save_to_file = False, country_2letter_iso_code = "IE")
# TRACKER = EmissionsTracker(project_name = "green_ml", save_to_file = False, measure_power_secs=10)

"""
Experiment
"""
class Experiment:

   def __init__(self, model_name, X, y, subsampling = False, feature_reduction = False, random_state = 0) -> None: # CHANGE random_state TO None
      self.model_name = model_name
      self.X = X
      self.y = y
      self.n_folds = N_FOLDS
      self.n_repeats = N_REPEATS
      self.random_state = random_state
      self.subsampling = subsampling
      self.subsamplig_frac = 0.2
      self.feature_reduction = feature_reduction
      self.results = []

   def do_subsampling(self, X_train:np.ndarray, y_train: np.ndarray):
      if self.subsampling:
         X_train_sub, y_train_sub = [], []
         for label in range(2): # 2 cause we are doing binary classification
            X_train_label = X_train[y_train == label]
            n_samples = round(X_train_label.shape[0] * self.subsamplig_frac)
            index = np.random.choice(X_train_label.shape[0], n_samples, replace=False)
            X_train_sub.extend([x for i, x in enumerate(X_train_label) if i in index])
            y_train_sub.extend([label] * n_samples)
         X_train_pp, y_train_pp = np.array(X_train_sub), np.array(y_train_sub)
      return X_train_pp, y_train_pp

   def do_pca(self, X_train: np.ndarray, X_test: np.ndarray):
      pca = PCA(0.95, random_state=self.random_state)
      X_train_pp = pca.fit_transform(X_train)
      X_test_pp = pca.transform(X_test)
      return X_train_pp, X_test_pp
   
   def do_data_scaling(self, X_train:np.ndarray, X_test:np.ndarray):
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      return X_train_scaled, X_test_scaled

   def preprocess_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray):
      if self.subsampling:
         X_train_pp, y_train_pp = self.do_subsampling(X_train, y_train)
      else:
         X_train_pp, y_train_pp = X_train, y_train
      if self.feature_reduction:
         X_train_pp, X_test_pp = self.do_pca(X_train_pp, X_test)
      else:
         X_test_pp = X_test
      X_train_scaled, X_test_scaled = self.do_data_scaling(X_train_pp, X_test_pp)
      return X_train_scaled, X_test_scaled, y_train_pp
    
   def new_model(self):
      if self.model_name == 'LogisticRegression':
         return LogisticRegression(random_state=self.random_state)
      if self.model_name == 'SVC':
         return SVC(random_state=self.random_state)
      elif self.model_name == 'RandomForestClassifier':
         return RandomForestClassifier(random_state=self.random_state)
      elif self.model_name == 'GradientBoostingClassifier':
         return GradientBoostingClassifier(random_state=self.random_state)
      elif self.model_name == 'DeepNeuralNetwork':
         return KDNN_v5.kdnn_model()
      else:
         print('ML model not supported')
   
   def run(self):
      TRACKER.start()
      # Repeated K-Fold Cross Validation
      rkf = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=self.random_state)
      for i, (train_index, test_index) in enumerate(rkf.split(self.X, self.y)):
         # Split data
         X_train, X_test = self.X[train_index], self.X[test_index]
         y_train, y_test = self.y[train_index], self.y[test_index]

         y_train = np.squeeze(y_train, axis=1)
         y_test = np.squeeze(y_test, axis=1)


         # Preprocess data
         start = time.time()
         # track = TRACKER.start()
         track_start = TRACKER.stop()
         X_train_pp, X_test_pp, y_train_pp = self.preprocess_data(X_train, X_test, y_train)
         preprocess_time = time.time() - start
         emissions_prep = TRACKER.stop() - track_start

         X_test_pp2 = np.copy(X_test_pp)
         while X_test_pp2.shape[0] < 1000:
            X_test_pp2 = np.concatenate([X_test_pp2, X_test_pp])


         # Train model
         self.model = self.new_model()

         if self.model_name == 'DeepNeuralNetwork': # Separeted to avoid printing training information
            # Train model
            start = time.time()
            # track = TRACKER.start()
            track_start = TRACKER.stop()
            self.model.fit(X_train_pp, y_train_pp, verbose=False)
            training_time = time.time() - start
            emissions_train = TRACKER.stop() - track_start
            # Evaluate model
            y_pred = self.model.predict(X_test_pp, verbose=False)
            # Prediction time (1000 samples)
            start = time.time()
            # track = TRACKER.start()
            track_start = TRACKER.stop()
            _ = self.model.predict(X_test_pp2[:1000, :], verbose=False)
            prediction_time = time.time() - start
            emissions_pred = TRACKER.stop() - track_start
         else:
            # Train model
            start = time.time()
            # track = TRACKER.start()
            track_start = TRACKER.stop()
            self.model.fit(X_train_pp, y_train_pp)
            training_time = time.time() - start
            emissions_train = TRACKER.stop() - track_start
            # Evaluate model
            y_pred = self.model.predict(X_test_pp)
            # Prediction time (1000 samples)
            start = time.time()
            # track = TRACKER.start()
            track_start = TRACKER.stop()
            _ = self.model.predict(X_test_pp2[:1000, :])
            prediction_time = time.time() - start
            emissions_pred = TRACKER.stop() - track_start

         #ROC curve parameters: false positive rate (fpr) and true positive rate (tpr)
         fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
         #AUC
         auc = metrics.auc(fpr, tpr)

         # Store results
         measures = {
               'accuracy': accuracy_score(y_test, np.round(y_pred)),
               'log_loss': log_loss(y_test, y_pred),
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc,
               'preprocess_time': preprocess_time,
               'training_time': training_time,
               'prediction_time': prediction_time,
               'model_size': sklearn_sizeof(self.model),
               'n_samples': X_train_pp.shape[0],
               'n_features': X_train_pp.shape[1],
               'emissions_prep': emissions_prep,
               'emissions_train': emissions_train,
               'emissions_pred': emissions_pred,
         }
         self.results.append(measures)

         del self.model

      return self.results
   
   def get_mesaure(self, measure):
      return [x[measure] for x in self.results]
