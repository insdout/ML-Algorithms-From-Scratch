import numpy as np
from svd import svd


def inv(M: np.ndarray) -> np.ndarray:
    # A^(-1) = (U@S@V.T)^(-1) = (V@S^(-1)@U.T)
    S, U, V = svd(M.copy())
    S_inv = 1 / S * np.eye(len(S))
    A_inv = V.T @ S_inv @ U.T
    return A_inv


if __name__ == '__main__':
    A = np.array([
        [6, 1, 1],
        [4, -2, 5],
        [2, 8, 7]
        ]).astype('float')

    print(inv(A))
    print()
    print(np.linalg.inv(A))
