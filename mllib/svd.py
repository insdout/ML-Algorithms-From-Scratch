import numpy as np


def power_iteration(A: np.ndarray, tolerance: float = 1e-15, max_iter: int = 1000) -> tuple[np.ndarray]:
    """
    Perform power iteration to find the dominant singular vector of the square matrix A.

    Parameters:
    - A (numpy.ndarray): Input square matrix.
    - tolerance (float): Tolerance for convergence, default is 1e-15.
    - max_iter (int): Maximum number of iterations, default is 1000.

    Returns:
    - numpy.ndarray: Dominant singular vector of A.
    """
    M = A.T @ A
    n, m = M.shape
    v = np.random.rand(m)
    v /= np.linalg.norm(v)
    i = 0
    while i < max_iter:
        w = M @ v
        w /= np.linalg.norm(w)
        delta = np.linalg.norm(w - v)
        print(i, delta)
        v = w
        i += 1
        if abs(delta) < tolerance:
            return w


def svd(A: np.ndarray) -> tuple[np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) on the input matrix A.

    Parameters:
    - A (numpy.ndarray): Input matrix.

    Returns:
    - tuple(numpy.ndarray): Tuple containing singular values, left singular vectors, and right singular vectors.
    """
    m, n = A.shape
    dim = min(m, n)
    u = []
    v = []
    sigma = []
    for i in range(dim):
        M = A.copy()
        for i in range(len(sigma)):
            M -= sigma[i]*np.outer(u[i], v[i])
        x = power_iteration(M)
        v_i = x/np.linalg.norm(x)
        sigma_i = np.linalg.norm(A @ v_i)
        u_i = (A @ v_i)/sigma_i
        v.append(v_i)
        u.append(u_i)
        sigma.append(sigma_i)
    return np.array(sigma), np.stack(u).T, np.stack(v)


if __name__ == '__main__':
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
    #M = np.array([[1, 1, 0],[1, 0, 1],[0, 1, 1]]).astype(float)
    # v1 = svd_1d(movieRatings)
    # print(v1)

    S, U, V = svd(M)
    U2, S2, V2 = np.linalg.svd(M, full_matrices=False)
    print('S:\n==================')
    print(S)
    print()
    print(S2)
    print('U:\n==================')
    print(U)
    print()
    print(U2)
    print('V:\n==================')
    print(V)
    print()
    print(V2)
    print('A:\n==================')
    print(U@(S*np.eye(len(S)))@V)
    print(U2@(S2*np.eye(len(S2)))@V2)
    print()
    print(U@U.T)
    print()
    print(V@V.T)