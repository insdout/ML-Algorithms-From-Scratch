import numpy as np
from copy import deepcopy


def qr_decomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass


def svd_decomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass


def power_iteration(M: np.ndarray, n_iter: int = 1000, tolerance: float = 1e-5) -> tuple[float, np.ndarray]:
    n, m = M.shape
    assert n == m , 'Matrix should be square!'
    v = np.random.rand(m)
    v = v/np.sqrt(v.T @ v)
    eps = float('inf')
    i = 0
    while eps > tolerance and i < n_iter:
        w = M @ v
        mu = (v.T @ M @ w) / (v.T @ w)
        eps = np.linalg.norm(M @ v - mu * v)
        v = w/np.sqrt(w.T @ w)
        i += 1
    return mu, v


def eigen_decomposition(M: np.ndarray, solver: str = 'power', tolerance: float = 1e-6) -> tuple[list, list]:
    n, m = M.shape
    assert n == m , 'Matrix should be square!'
    eigenvalues = []
    eigenvectors = []
    eigenvectors2 = []
    if solver == 'power':
        N = deepcopy(M)
        for _ in range(n):
            eigenvalue, eigenvector = power_iteration(N, tolerance=tolerance)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            eigenvectors2.append(find_eigenvector_safe(M, eigenvalue))
            N -= eigenvalue * np.outer(eigenvector, eigenvector)
    return eigenvalues, np.array(eigenvectors2).T


def find_eigenvector_safe(M, eigenvalue):
    A = deepcopy(M)
    n = A.shape[0]
    I = np.eye(n)
    eigenvector = np.linalg.svd(A - eigenvalue * I)[2][-1].T
    # Normalize the eigenvector
    eigenvector /= np.linalg.norm(eigenvector)
    return eigenvector


if __name__ == '__main__':
    from numpy.linalg import eig
    M = np.array([[5, 2, 0], [2, 5, 0], [-3, 4, 6]]).astype(float)
    M = np.array([[1, 2, 1], [1, 7, 3], [1, 2, 6]]).astype(float)
    eigenvalues, eigenvectors = eigen_decomposition(M)
    eigenvalues2, eigenvectors2 = eig(M)
    print('\nEigenvalues:\n===================')
    print(list(map(lambda x: round(x, 3), eigenvalues)))
    idx = np.argsort(eigenvalues2)[::-1]
    print(list(map(lambda x: round(x, 3), eigenvalues2[idx])))
    print('\nEigenvectors:\n===================')
    for i in range(eigenvectors2.shape[1]):
        print(f'Eigenvectors corresponding to eigenvalue {eigenvalues[i]:3.3f}:')
        print(eigenvectors[:, i])
        print(eigenvectors2[:, idx[i]])
        print()
    print('\nSanity Check:\n===================')
    for i in range(len(eigenvalues)):
        print('Eigenvector:')
        print(eigenvectors[:, i] )
        print('lambda * eigenvector:')
        print(eigenvectors[:, i] * eigenvalues[i])
        print('M @ eigenvector:')
        print(M@eigenvectors[:, i])
        print()
    print('===================')
    print('===================')

