import numpy as np
from numpy.linalg import norm
import copy

# https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
# https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/


class ConvergenceError(Exception):
    pass


def svd_1d(A, epsilon=1e-10, max_iter=1000):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = np.random.rand(min(n, m))
    x /= norm(x)
    lastV = None
    currentV = x

    if n > m:
        B = A.T @ A
    else:
        B = A @ A.T

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV
        if iterations > max_iter:
            raise ConvergenceError(f'Didn\'t converge in {iterations}!')


def svd(B, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = copy.deepcopy(B)
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs


if __name__ == "__main__":
    M = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')

    M = np.array([[1, 2, 1], [1, 7, 3], [1, 2, 6]]).astype(float)
    # v1 = svd_1d(movieRatings)
    # print(v1)

    S, U, V = svd(M)
    U2, S2, V2 = np.linalg.svd(M, full_matrices=False)
    print('==================')
    print(S)
    print()
    print(S2)
    print('==================')
    print(U)
    print()
    print(U2)
    print('==================')
    print(-V)
    print()
    print(V2)

