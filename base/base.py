import numpy as np
from collections import defaultdict

class Estimator:
    y_required = True
    fit_required = True

    def fit(self, X, y=None):
        self._check_input(X, y)
        self._fit(X, y)

    def predict(self, X):
        self._check_input(X)
        return self._predict(X)

    def _check_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 0:
            raise ValueError("The array X must be non-empty")
        elif X.ndim > 2:
            raise ValueError("Input must be a 2-dimensional array.")
        elif X.ndim == 1:
            X = np.expand_dims(X, axis=1)

        if self.y_required:
            if not y:
                raise ValueError("Argument y is required.")
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if y.ndim == 0:
                raise ValueError("The array y must be non-empty")
            elif y.ndim == 1:
                y = np.expand_dims(y, axis=1)
        return X, y 

    
    def _fit(self, X, y=None):
        raise NotImplementedError("Subclasses must implement _fit method.")

    def _predict(self, X):
        raise NotImplementedError("Subclasses must implement _predict method.") 
    

class BaseOptimizer:
    def __init__(self, gradient_fn, learning_rate, tolerance=1e-5, max_iter=1000):
        self.learning_rate = learning_rate
        self.gradient_fn = gradient_fn
        self.tol = tolerance
        self.max_iter = max_iter
        self.history = defaultdict(list)
    
    def update_parameters(self, parameters, gradient_fn):
        raise NotImplementedError("Subclasses must implement update_parameters method.")
    
    def optimize(self):
        raise NotImplementedError("Subclasses must implement optimize method.")



    