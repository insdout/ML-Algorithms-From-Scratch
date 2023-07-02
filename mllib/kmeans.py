import numpy as np
import random
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
    x = np.asanyarray(x)
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
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)

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
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)

    diff = x1 - x2
    distance = norm(diff, p=2)
    return distance


class KMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iters=1000, init="random"):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.y_required = False
        self.centroids = []

    def _init_centroids(self, X):
        if self.init == 'random':
            centroids = random.sample(
                X, 
                self.n_clusters
                )
        elif self.init == 'kmeans++':
            centroids = [np.random.choice(X)]

    def _fit(self, X, y=None):
        pass

    def _predict(self, X):
        pass

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
    print(random.sample(rand_sample, 3))