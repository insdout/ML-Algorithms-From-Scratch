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
