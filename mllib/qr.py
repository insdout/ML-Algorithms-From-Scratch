import numpy as np 
from copy import copy


def proja_b(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the projection of vector b onto vector a.

    Parameters:
    - a (numpy.ndarray): Vector a.
    - b (numpy.ndarray): Vector b.

    Returns:
    - float: Projection of vector b onto vector a.
    """
    proj = np.dot(a, b)/np.dot(a, a)
    return proj


def norm(a: np.ndarray) -> float:
    """
    Calculates the Euclidean norm of a vector.

    Parameters:
    - a (numpy.ndarray): Vector.

    Returns:
    - float: Euclidean norm of the vector.
    """
    norm = np.sqrt(np.sum(np.dot(a, a)))
    return norm


def QR(A: np.ndarray) -> tuple[np.ndarray]:
    """
    Performs QR decomposition on a matrix A.

    Parameters:
    - A (numpy.ndarray): Input matrix.

    Returns:
    - tuple[numpy.ndarray]: Tuple containing Q and R matrices of the QR decomposition.
    """
    M = copy(A).astype('float')
    a1 = M[:, 0]
    q = [a1/norm(a1)]
    print(f'q: {q[-1]}')
    for col in range(1, M.shape[1]):
        an = M[:, col]
        for k in range(len(q)):
            an -= proja_b(q[k], an)*q[k]
        q.append(an/norm(an))
        print(f'q: {q[-1]}')
    Q = np.stack(q).T
    R = Q.T @ copy(A).astype('float')
    return Q, R


if __name__ == '__main__':
    a = np.array([1, 0])
    b = np.array([0, 1])
    c = np.array([1, 1])

    print(proja_b(a, a))
    print(proja_b(a, b))
    print(proja_b(c, b))
    print(proja_b(b, c))

    A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]])
    q, r = QR(A)
    print(q)
    print()
    print(r)
    print()
    print(q.T@q)
    print()
    print(q@r)
