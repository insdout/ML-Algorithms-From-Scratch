from typing import List, Optional
import numpy as np
from base import BaseEstimator
from utils import euclidean_distance

class KMeans(BaseEstimator):
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 1000,
        init: str = "random",
        tolerance: float = 0.0001
    ) -> None:
        """
        KMeans clustering algorithm.

        Parameters:
        - n_clusters (int): Number of clusters.
        - max_iters (int): Maximum number of iterations.
        - init (str): Initialization method, either "random" or "kmeans++".
        - tolerance (float): Convergence tolerance.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.tolerance = tolerance
        self.y_required = False
        self.centroids: List[np.ndarray] = []

    def _init_centroids(self, X: np.ndarray) -> None:
        """
        Initialize cluster centroids based on the specified method.

        Parameters:
        - X (np.ndarray): Input data.
        """
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

    def _find_next_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Find the next cluster centroid using the kmeans++ initialization.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Next cluster centroid.
        """
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

    def _min_distance_from_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the mean distance from each point to the cluster centroids.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Mean distances from each point to the centroids.
        """
        distances = self._distance_from_centroids(X)
        min_distances = np.mean(distances, axis=0)
        return np.asarray(min_distances)

    def _distance_from_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the distances from each point to each cluster centroid.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Array of distances.
        """
        distance = []
        for x in X:
            row_distances = []
            for c in self.centroids:
                row_distances.append(euclidean_distance(x, c))
            distance.append(row_distances)
        return np.array(distance)

    def _closest_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Determine the index of the closest centroid for each point.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Index of the closest centroid for each point.
        """
        distances = self._distance_from_centroids(X)
        print(f"distances: {distances}")
        closest_centroids = np.argmin(distances, axis=1)
        print(f"closest centroids: {closest_centroids}")
        return closest_centroids

    def _get_new_centroids(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Calculate new centroids based on assigned datapoints.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - List[np.ndarray]: List of new centroids.
        """
        closest_centroids = self._closest_centroids(X)
        new_centroids = []
        for i in range(self.n_clusters):
            assigned_idx = closest_centroids == i
            print(f"assigned indexes: {assigned_idx}")
            assigned_datapoints = X[assigned_idx]
            print(f"cluster {i} new centroid: {np.mean(assigned_datapoints, axis=0)}")
            new_centroids.append(np.mean(assigned_datapoints, axis=0))
        return new_centroids

    def _is_converged(self, old_centroids: List[np.ndarray], new_centroids: List[np.ndarray]) -> bool:
        """
        Check if the algorithm has converged.

        Parameters:
        - old_centroids (List[np.ndarray]): Old centroids.
        - new_centroids (List[np.ndarray]): New centroids.

        Returns:
        - bool: True if converged, False otherwise.
        """
        total_dist = 0
        for old_centroid, new_centroid in zip(old_centroids, new_centroids):
            total_dist += euclidean_distance(old_centroid, new_centroid)
        return total_dist < self.tolerance

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the KMeans model to the data.

        Parameters:
        - X (np.ndarray): Input data.
        - y (Optional[np.ndarray]): Not used.

        Returns:
        - None
        """
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

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each data point.

        Parameters:
        - X (np.ndarray): Input data.

        Returns:
        - np.ndarray: Predicted cluster labels.
        """
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

    def get_accuracy_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy of the clustering result.

        Parameters:
        - y_true (np.ndarray): True cluster labels.
        - y_pred (np.ndarray): Predicted cluster labels.

        Returns:
        - float: Accuracy score.
        """
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

    print(f'Accuracy: {get_accuracy_clustering(y, pred)}')