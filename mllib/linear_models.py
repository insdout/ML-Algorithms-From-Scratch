from base import BaseEstimator
from sgd import SGD
import numpy as np


class LinearRegression(BaseEstimator):
    def __init__(self, solver="closed_form", C=0, max_iter=1000):
        self.solver = solver
        self.C = C
        self.max_iter = max_iter

    def loss_fn(self, X_intercept, y, parameters):
        n = y.shape[0]
        return (1/(2*n))*(np.sum((X_intercept @ parameters - y)**2 + np.sum(self.C * np.concatenate(([0], parameters[1:]))**2)))

    def gradient_fn(self, X_intercept, y, parameters):
        n = y.shape[0]
        return (1/(2*n))*(X_intercept.T @ (X_intercept @ parameters - y) + self.C * np.concatenate(([0], parameters[1:])))

    def closed_form(self, X_intercept, y, parameters):
        I = np.eye(parameters.shape[0])
        I[0,0] = 0
        return np.linalg.inv(X_intercept.T @ X_intercept + self.C*I) @ X_intercept.T @ y

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit(self, X, y):
        X, y = self._check_input(X, y)
        X_intercept = self._add_intercept(X)
        n_parameters = X_intercept.shape[1]
        self.parameters = np.random.rand(n_parameters)
        if self.solver == "closed_form":
            self.parameters = self.closed_form(X_intercept, y, self.parameters)
        elif self.solver == "sgd":
            self.sgd = SGD(
                self.gradient_fn, 
                self.parameters, 
                1e-1, 
                self.loss_fn, 
                batch_size=y.size, 
                max_iter=self.max_iter, 
                tolerance=1e-2
                )
            self.parameters = self.sgd.optimize(X_intercept, y)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=10, n_features=2, noise=1, random_state=42)
    lr = LinearRegression(solver="sgd", C=1, max_iter=1000)
    lr.fit(X, y)
    print(lr.parameters)
    lr2 = LinearRegression(solver="closed_form", C=1, max_iter=1000)
    lr2.fit(X, y)
    print(lr2.parameters)