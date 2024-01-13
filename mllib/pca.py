import numpy as np
from numpy.linalg import eig
from scipy.linalg import svd
from base import BaseEstimator


class PCA:

    def __init__(self, num_components):
        self.num_components = num_components
        self.components = None
        self.mean = None
    
    def _standardize(self, X):
        return X - self.mean

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_standardized = self._standardize(X)
        
        covariance_matrix = X_standardized.T @ X_standardized /(X.shape[0]-1)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Sort eigenvectors by eigenvalues in descending order
        sorted_idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_idxs[:self.num_components]]

    def transform(self, X):
        if self.components is None:
            raise Exception("Call fit method first!")

        X_standardized = self._standardize(X)
        return np.dot(X_standardized, self.components)

    def fit_transform(self, X):
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

    # Using sklearn PCA for comparison
    sk_pca = PCA(num_components=2)
    my_result = sk_pca.fit_transform(X)


