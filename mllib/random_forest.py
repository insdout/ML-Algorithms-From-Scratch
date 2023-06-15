import numpy as np
from base import BaseEstimator
from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators, criterion, max_depth, max_features, max_samples_split, regression):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples_split = max_samples_split
        self.regression = regression

    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass
