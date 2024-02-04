import numpy as np
from base import BaseEstimator
from utils import minkowski_distance, cosine_distance


class KNN(BaseEstimator):
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        if metric == "minkowski":
            self.distance = minkowski_distance
        elif metric == "cosine":
            self.distance = cosine_distance
        else:
            raise ValueError(f"{metric} is not supported.")

    def _fit(self, X, y):
        self.X = X
        self.y = y
        self.fit_required = False

    def _predict(self, X):
        predictions = [self._predict_row(x) for x in X]
        return np.array(predictions)
    
    def _predict_row(self, x):
        n = self.n_neighbors
        distances = self._get_distances(x)
        n_nearest_ind = np.argpartition(distances, n)[:n]
        n_nearest_y = self.y[n_nearest_ind]
        predictions_row = self. _get_neighbors_votes(n_nearest_y)
        return predictions_row
    
    def _get_distances(self, x):
        distances = [self.distance(x, point) for point in self.X]
        return np.array(distances)

    def _get_neighbors_votes(self, y):
        raise NotImplementedError("Subclass must implement _get_neighbors_votes method.")


class KNNRegressor(KNN):
    def __init__(self, n_neighbors=3, metric="minkowski"):
        super().__init__(n_neighbors=n_neighbors, metric=metric)
    
    def _get_neighbors_votes(self, y):
        return np.mean(y)
    

class KNNClassifier(KNN):
    def __init__(self, n_neighbors=3, metric="minkowski"):
        super().__init__(n_neighbors=n_neighbors, metric=metric)
    
    def _get_neighbors_votes(self, y):
        counts = np.bincount(y)
        most_frequent_class = np.argmax(counts)
        return most_frequent_class
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X, y = make_classification(
        n_features=20, 
        n_redundant=12, 
        n_informative=5, 
        random_state=42, 
        n_clusters_per_class=1, 
        class_sep=0.5, 
        n_classes=3
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.33, 
        random_state=42
        )
    knn = KNNClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print()
