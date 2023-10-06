# Machine learning pipeline libraries
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

# Metrics libraries
import time
from sklearn.metrics import accuracy_score
import types

class Experiment:

    def __init__(self, model, X, y) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.n_folds = 10
        self.n_repeats = 2
        self.random_state = 0
        self.results = []

    def preprocess_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def run(self):
        # Repeated K-Fold Cross Validation
        rkf = RepeatedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=self.random_state)
        for i, (train_index, test_index) in enumerate(rkf.split(self.X)):

            # Split data
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Preprocess data
            X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)

            # Train model
            start = time.time()
            self.model.fit(X_train_scaled, y_train)
            training_time = time.time() - start

            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)

            # Store results
            measures = {
                'accuracy': accuracy_score(y_test, y_pred),
                'training_time': training_time,
                'model_size': sklearn_sizeof(self.model)
            }
            self.results.append(measures)

        return self.results
    
    def get_mesaure(self, measure):
        return [x[measure] for x in self.results]

    

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
