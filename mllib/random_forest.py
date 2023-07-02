import numpy as np
from base import BaseEstimator
from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators, criterion, max_depth, max_features, min_samples_split, regression):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.regression = regression
        self.estimators = []
     
    def _fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)
        self.fit_required = False

    def predict(self, X):
        predictions = self._predict(X)
        return np.array(predictions)
    

class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, max_features="auto", min_samples_split=2):
        super().__init__(n_estimators, criterion, max_depth, max_features, min_samples_split, regression=False)

        if self.max_features == "auto":
            tree_max_features = "sqrt"
        else:
            tree_max_features = self.max_features
            
        for _ in range(self.n_estimators):
            self.estimators.append(
                DecisionTreeClassifier(
                    criterion=self.criterion, 
                    max_depth=None, 
                    max_features=tree_max_features, 
                    min_samples_split=self.min_samples_split
                    )
                )

    def _predict(self, X):
        row_prediction = []
        for estimator in self.estimators:
            row_prediction.append(estimator.predict(X))
        row_prediction = np.stack(row_prediction, axis=1)
        counts = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=row_prediction)
        return counts
    

class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=100, criterion="mse", max_depth=None, max_features="auto", min_samples_split=2):
        super().__init__(n_estimators, criterion, max_depth, max_features, min_samples_split, regression=True)

        if self.max_features == "auto":
            tree_max_features = "div3"
        else:
            tree_max_features = self.max_features
            
        for _ in range(self.n_estimators):
            self.estimators.append(
                DecisionTreeRegressor(
                    criterion=self.criterion, 
                    max_depth=None, 
                    max_features=tree_max_features, 
                    min_samples_split=self.min_samples_split
                    )
                )

    def _predict(self, X):
        # TODO: Check correctness of RandomForestRegressor
        row_prediction = []
        for estimator in self.estimators:
            row_prediction.append(estimator.predict(X))
        return np.mean(row_prediction, axis=0)

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X, y = make_classification(
        n_features=20, n_redundant=2, n_informative=15, random_state=42, n_clusters_per_class=1, class_sep=2, n_classes=3
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print()

    X_train, y_train = make_regression(
        n_features=6, n_informative=4, random_state=1)
