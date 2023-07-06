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
        w = np.asarray(w)
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


def cosine_similarity(x1, x2):
    """
    Compute the cosine similarity between two points.

    D(x1, x2) = dot(x1, x2)/(|x1|*|x2|)

    Args:
        x1: First point (array-like object).
        x2: Second point (array-like object).

    Returns:
        Cosine similarity between x1 and x2.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    similarity = np.dot(x1, x2)/(norm(x1)*norm(x2))
    return similarity


def cosine_distance(x1, x2):
    """
    Compute the cosine distance between two points.

    D(x1, x2) = 1 - dot(x1, x2)/(|x1|*|x2|)

    Args:
        x1: First point (array-like object).
        x2: Second point (array-like object).

    Returns:
        Cosine distance between x1 and x2.
    """
    similarity = cosine_similarity(x1, x2)
    return 1 - similarity


if __name__ == "__main__":
    from scipy.spatial import distance
    from numpy import linalg as LA
    
    x1 = [1, 2, 3]
    x2 = [4, 5, 6]
    w = [1, 2, 4]
    tol = 1e-5

    d1 = norm(x1,  p=2)
    d2 = LA.norm(x1, ord=2)
    assert abs(d1 - d2) < tol, "Norm. Difference is too high."

    d1 = minkowski_distance(x1, x2, w=w, p=3)
    d2 = distance.minkowski(x1, x2, w=w, p=3)
    assert abs(d1 - d2) < tol, "Minkowski distance. Difference is too high."

    d1 = euclidean_distance(x1, x2)
    d2 = distance.euclidean(x1, x2)
    assert abs(d1 - d2) < tol, "Euclidean Distance. Difference is too high."

    d1 = cosine_distance(x1, x2)
    d2 = distance.cosine(x1, x2)
    assert abs(d1 - d2) < tol, "Cosine Distance. Difference is too high."

    print("All tests passd!")