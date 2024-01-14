import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) class.

    Parameters:
    - n_components (int): Number of principal components to retain.

    Attributes:
    - n_components (int): Number of principal components to retain.
    - components (ndarray or None): Principal components obtained after fitting.
    - mean (ndarray or None): Mean of the input data used for standardization.
    """

    def __init__(self, n_components: int):
        """
        Initialize PCA with the specified number of components.

        Parameters:
        - n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input data.

        Parameters:
        - X (ndarray): Input data.

        Returns:
        - ndarray: Standardized data.
        """
        return X - self.mean

    def _swap_signs(self, M: np.ndarray) -> np.ndarray:
        """
        Swap signs of the columns of the matrix based on the column with maximum absolute value.

        Parameters:
        - M (ndarray): Input matrix.

        Returns:
        - ndarray: Matrix with signs swapped.
        """
        max_abs_cols = np.argmax(np.abs(M), axis=0)
        signs = np.sign(M[max_abs_cols, range(M.shape[1])])
        M *= signs
        return M

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the input data.

        Parameters:
        - X (ndarray): Input data.
        """
        self.mean = np.mean(X, axis=0)
        X_standardized = self._standardize(X)

        S = X_standardized.T @ X_standardized / (X.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eig(S)
        # Sort eigenvectors by eigenvalues in descending order
        sorted_idxs = np.argsort(eigenvalues)[::-1]
        self.components = self._swap_signs(eigenvectors[:, sorted_idxs[:self.n_components]])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted PCA model.

        Parameters:
        - X (ndarray): Input data.

        Returns:
        - ndarray: Transformed data.
        """
        if self.components is None:
            raise Exception("Call fit method first!")

        X_standardized = self._standardize(X)
        return np.dot(X_standardized, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model to the input data and transform it.

        Parameters:
        - X (ndarray): Input data.

        Returns:
        - ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':

    # Testing the implementation
    X = np.array([
        [1.5, 2.0, 3.5, 4.2, 5.8],
        [2.0, 3.2, 4.5, 5.7, 6.2],
        [3.2, 4.5, 5.1, 6.0, 7.5],
        [4.1, 5.6, 6.9, 7.2, 8.1],
        [5.3, 6.4, 7.7, 8.4, 9.2],
        [6.2, 7.1, 8.4, 9.3, 10.0],
        [7.5, 8.3, 9.6, 10.2, 11.0],
        [8.8, 9.6, 10.9, 11.5, 12.2],
        [9.9, 10.8, 11.7, 12.3, 13.0],
        [10.5, 11.7, 12.9, 13.6, 14.5]
    ]).astype(float)

    # Msing sklearn PCA for comparison
    pca = PCA(n_components=2)
    my_result = pca.fit_transform(X)
    print(my_result @ pca.components.T + pca.mean)


