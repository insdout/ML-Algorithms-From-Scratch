import numpy as np
from base import BaseEstimator
from utils import euclidean_distance


class KMeans(BaseEstimator):
    def __init__(
            self,
            n_clusters,
            max_iters=1000,
            init="random",
            tolerance=0.0001
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.tolerance = tolerance
        self.y_required = False
        self.centroids = []

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        if self.init == 'random':
            indexes = np.random.choice(
                n_samples,
                size=self.n_clusters,
                replace=False
            )
            print(f"centroid indexes: {indexes}")
            self.centroids = [X[ind] for ind in indexes]
            print(f"centroids: {self.centroids}")
        elif self.init == 'kmeans++':
            index = np.random.randint(n_samples, size=1)
            self.centroids = [X[index]]
            while len(self.centroids) < self.n_clusters:
                self.centroids.append(self._find_next_cluster(X))
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def _find_next_cluster(self, X):
        n_samples = X.shape[0]
        min_distances = self._min_distance_from_centroids(X)
        squared_min_distances = min_distances**2
        probabilities = squared_min_distances/np.sum(squared_min_distances)
        new_centroid_indx = np.random.choice(
            n_samples,
            size=1,
            p=probabilities
        )
        return X[new_centroid_indx]

    def _min_distance_from_centroids(self, X):
        distances = self._distance_from_centroids(X)
        min_distances = np.mean(distances, axis=0)
        return np.asarray(min_distances)

    def _distance_from_centroids(self, X):
        distance = []
        for x in X:
            row_distances = []
            for c in self.centroids:
                row_distances.append(euclidean_distance(x, c))
            distance.append(row_distances)
        return np.array(distance)

    def _closest_centroids(self, X):
        distances = self._distance_from_centroids(X)
        print(f"distances: {distances}")
        closest_centroids = np.argmin(distances, axis=1)
        print(f"closest centroids: {closest_centroids}")
        return closest_centroids

    def _get_new_centroids(self, X):
        closest_centroids = self._closest_centroids(X)
        new_centroids = []
        for i in range(self.n_clusters):
            assigned_idx = closest_centroids == i
            print(f"assigned indexes: {assigned_idx}")
            assigned_datapoints = X[assigned_idx]
            print(f"cluster {i} new centroid: {np.mean(assigned_datapoints, axis=0)}")
            new_centroids.append(np.mean(assigned_datapoints, axis=0))
        return new_centroids

    def _is_converged(self, old_centroids, new_centroids):
        total_dist = 0
        for old_centroid, new_centroid in zip(old_centroids, new_centroids):
            total_dist += euclidean_distance(old_centroid, new_centroid)
        return total_dist < self.tolerance

    def _fit(self, X, y=None):
        self._init_centroids(X)
        print(f"centroids: {self.centroids}")
        for i in range(self.max_iters):
            old_centroids = self.centroids
            new_centroids = self._get_new_centroids(X)
            if self._is_converged(old_centroids, new_centroids):
                self.centroids = new_centroids
                print(f"converged on {i+1}th iteration!")
                break
            self.centroids = new_centroids
        self.fit_required = False

    def _predict(self, X):
        closest_centroids = self._closest_centroids(X)
        return np.array(closest_centroids)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(
        n_samples=100,
        centers=3,
        n_features=2,
        random_state=0
        )
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    pred = kmeans.predict(X)

    def get_accuracy_clustering(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        label_mapping = {}
        for c in np.unique(y_true):
            class_indexes = y_true == c
            counts = np.bincount(y_pred[class_indexes])
            most_frequent = np.argmax(counts)
            assert most_frequent not in label_mapping, f"{most_frequent} class is already used."
            label_mapping[most_frequent] = c

        mapped_y_pred = np.array(list(map(lambda x: label_mapping.get(x), y_pred)))

        return accuracy_score(y_true, mapped_y_pred)
        
    print(get_accuracy_clustering(y, pred))