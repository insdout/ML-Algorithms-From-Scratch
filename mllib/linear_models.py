from base import BaseEstimator
import numpy as np


class LinearRegression(BaseEstimator):
    def __init__(self, solver="pseudoinverse", C=0):
        self.solver = solver
        self.C = C

    def loss_fn(self, X_intercept, y, parameters):
        I = np.eye(parameters.shape[0])
        I[0,0] = 0
        n = y.shape[0]
        return (1/n) * (np.sum(X_intercept @ parameters - y)**2 + self.C * np.sum(parameters[1:]**2))
    
    def gradient_fn(self, X_intercept, y, parameters):
        pass

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit(self, X, y):
        X_intercept = self._add_intercept(X)
        n_parameters = X_intercept.shape[1]
        self.parameters = np.random.rand(n_parameters)
