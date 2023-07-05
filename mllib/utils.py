import numpy as np


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

    diff = x1 - x2
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

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score
    from scipy.spatial import distance
    import random

    point1 = [1, 2, 3]
    point2 = [4, 5, 6]
    dist = minkowski_distance(point1, point2, p=3)
    print(dist)

    
    print(distance.minkowski(point1, point2, p=3))

    print(f"minkowski p=2:"
          f"{minkowski_distance(point1, point2, p=2)}"
          f"euclidean: {euclidean_distance(point1, point2)}")