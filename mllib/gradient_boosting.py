from base import BaseEstimator
from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor
import numpy as np

class GradientBoosting(BaseEstimator):
    def __init__(self, loss_fn, loss_grad_fn, criterion, n_estimators, max_depth, max_features, min_samples_split):
        self.loss_fn = loss_fn
        self.loss_grad_fn = loss_grad_fn
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samplees_split = min_samples_split
        self.n_estimators = n_estimators

    def _fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


if __name__ == "__main__":
    pass
