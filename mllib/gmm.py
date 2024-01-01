from base import BaseEstimator
import numpy as np


class GMM(BaseEstimator):
    def __init__(self, n):
        self.n = n
        self.means = []
        self.simas = []

    @staticmethod
    def pdf_multivariate(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        d = len(mu)
        denomenator = ((2*np.pi)**(d/2)) * (np.linalg.det(sigma)**(1/2))
        numenator = np.exp(-(1/2)*(x - mu).T @ sigma @ (x - mu))
        return numenator/denomenator

    def e_step(self):
        pass

    def m_step(self):
        pass

    def _fit(self, X):
        pass

    def _predict(self):
        pass


if __name__ == "__main__":
    mu = np.array([0, 0])
    sigma = np.eye(2)
    x = np.array([0, 0])
    p = GMM.pdf_multivariate(x, mu, sigma)
    print(f'PDF: {p}')
