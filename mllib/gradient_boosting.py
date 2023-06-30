from base import BaseEstimator
from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor
import numpy as np


class Loss:
    def gradient(self):
        raise NotImplementedError("Subclasses must implement predict method.")
    
    def hessian(self):
        raise NotImplementedError("Subclasses must implement predict method.")
    
    def raw_prediction_to_decision(self):
        raise NotImplementedError("Subclasses must implement predict method.")
    
    def gain(self):
        raise NotImplementedError("Subclasses must implement predict method.")
    
    

class GradientBoosting(BaseEstimator):
    def __init__(self, loss, criterion, learning_rate, n_estimators, max_depth, max_features, min_samples_split):
        self.loss = loss
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samplees_split = min_samples_split
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []

    def _initial_prediction(self, y):
        raise NotImplementedError("Subclasses must implement predict method.")

    def _fit(self, X, y):
        y_pred = self._initial_prediction(y)
        for estimator in self.estimators:
            residuals = self.loss.gradient(y, y_pred)
            estimator.fit(X, y)
            self.estimators.append(estimator)
            predictions = self.estimator.predict(X)
            y_pred += self.learning_rate * predictions

    def _predict(self, X):
        prediction = self._initial_prediction(X)
        for estimator in self.estimators:
            prediction += self.learning_rate * estimator.predict(X)
        prediction = self.loss.raw_prediction_to_decision(prediction)
        return prediction


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X, y = make_classification(
        n_features=20, n_redundant=2, n_informative=15, random_state=42, n_clusters_per_class=1, class_sep=2, n_classes=3
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = GradientBoostingClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print()


    X_train, y_train = make_regression(
        n_features=6, n_informative=4, random_state=1)
