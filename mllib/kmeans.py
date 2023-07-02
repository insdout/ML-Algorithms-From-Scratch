import numpy as np
from base import BaseEstimator

def norm(x, p=2):
    """
    Compute p norm of vector x

    Args:
        x1: First point (array-like object).
        x2: Second point (array-like object).
        p: Parameter value for the norm (default: 2).
    
    Returns:
        p-norm of vector x
    """
    x = np.asarray(x)
    return np.sum(np.abs(x**p))**(1/p)


def minkowski_distance(x1, x2, p=2, w=None):
    """
    Compute the Minkowski distance between two points.
    
    D(x1, x2) = (sum |x1 - x2|**p)**(1/p)

    Args:
        x1: First point (array-like object).
        x2: Second point (array-like object).
        p: Parameter value for the Minkowski distance (default: 2).
    
    Returns:
        Minkowski distance between x1 and x2.
    """
    if p < 1:
        raise ValueError("Parameter p must be greater or equal to 1.")
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    diff  = x1 - x2
    if w:
        diff = w**(1/p)*diff
    distance = norm(diff, p)
    return distance

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.
    
    D(x1, x2) = (sum |x1 - x2|**2)**(1/2)

    Args:
        x1: First point (array-like object).
        x2: Second point (array-like object).
    
    Returns:
        Euclidean distance between x1 and x2.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    diff = x1 - x2
    distance = norm(diff, p=2)
    return distance


class KMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iters=1000, init="random", tolerance=0.0001):
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
            self.centroids = [X[ind] for ind in indexes]
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
        new_centroid_indx = np.random.choice(n_samples, size=1, p=probabilities)
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
        closest_centroids = np.argmin(distances, axis=0)
        return closest_centroids
    
    def _get_new_centroids(self, X):
        closest_centroids = self._closest_centroids(X)
        new_centroids = []
        for i in range(self.n_clusters):
            assigned_idx = closest_centroids == i
            assigned_datapoints = X[assigned_idx]
            new_centroids.append(np.mean(assigned_datapoints, axis=1))
        return new_centroids
    
    def _is_converged(self, old_centroids, new_centroids):
        total_dist = 0
        for old_centroid, new_centroid in zip(old_centroids, new_centroids):
            total_dist += euclidean_distance(old_centroid, new_centroid)
        return total_dist < self.tolerance
    
    def _fit(self, X, y=None):
        self.centroids = self._init_centroids(X)
        for i in range(self.max_iters):
            old_centroids = self.centroids
            new_centroids = self._get_new_centroids(X)
            if self._is_converged(old_centroids, new_centroids):
                self.centroids = new_centroids
                break
            self.centroids = new_centroids
        self.fit_required = False

    def _predict(self, X):
        closest_centroids = self._closest_centroids(X)
        return np.array(closest_centroids)
    

if __name__ == "__main__":
    point1 = [1, 2, 3]
    point2 = [4, 5, 6]
    distance = minkowski_distance(point1, point2, p=3)
    print(distance)

    from scipy.spatial import distance
    print(distance.minkowski(point1, point2, p=3))

    print(f"minkowski p=2: {minkowski_distance(point1, point2, p=2)} euclidean: {euclidean_distance(point1, point2)}")

    import random
    rand_sample = np.random.rand(5,2)
    print(rand_sample)
    print(random.choice(rand_sample))