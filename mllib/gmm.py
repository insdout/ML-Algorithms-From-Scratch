from base import BaseEstimator
import numpy as np
from scipy.stats import wishart
from scipy.stats import multivariate_normal

class CovergenceError(Exception):
    pass


class GMM(BaseEstimator):
    y_required = False

    def __init__(self, n_components: int, max_iter: int = 1000, tolerance: float = 1e-5) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.means = []
        self.sigmas = []
        self.pi_c = np.ones(n_components)/n_components

    @staticmethod
    def pdf_multivariate(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        d = len(mu)
        denomenator = ((2*np.pi)**(d/2)) * (np.linalg.det(sigma)**(1/2))
        numenator = np.exp(-(1/2)*(x - mu).T @ np.linalg.inv(sigma) @ (x - mu))
        return numenator/denomenator

    def pdf_multivariate2(self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        d = len(mu)
        denominator = ((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2))
        # Calculate the exponent for each sample in x
        exponent = -0.5 * np.sum(np.dot(x - mu, np.linalg.inv(sigma)) * (x - mu), axis=1)
        # Calculate the numerator for each sample in x
        numerator = np.exp(exponent)
        # Calculate the probability density for each sample in x
        pdf_values = numerator / denominator
        return pdf_values

    def _init_params(self, n_components: int, n_features: int) -> None:
        self.means = np.random.rand(n_components, n_features)
        sigmas = [wishart.rvs(df=n_features, scale=np.eye(n_features)) for _ in range(n_components)]
        self.sigmas = np.stack(sigmas, axis=0)

    def _get_likelihoods(self, x: np.ndarray) -> np.ndarray:
        likelihoods = []
        for c in range(self.n_components):
            likelihoods.append(self.pdf_multivariate2(x, mu=self.means[c], sigma=self.sigmas[c]))
        return np.stack(likelihoods, axis=1)
    
    def _E_step(self, x):
        n_samples = x.shape[0]
        gamma_i_c = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            for c in range(self.n_components):
                gamma_i_c[i, c] = self.pdf_multivariate(x[i], mu=self.means[c], sigma=self.sigmas[c])*self.pi_c[c]
                #gamma_i_c[i, c] = multivariate_normal.pdf(x[i], mean=self.means[c], cov=self.sigmas[c])*self.pi_c[c]
            gamma_i_c[i, :] = gamma_i_c[i, :]/np.sum(gamma_i_c[i, :])
        self.gamma = gamma_i_c
        return gamma_i_c

    def _E_step2(self, x):
            likelihoods = self._get_likelihoods(x)
            weighted_likelihoods = likelihoods*self.pi_c
            gamma_i_c = weighted_likelihoods/np.sum(weighted_likelihoods, axis=1).reshape(-1, 1)
            self.gamma = gamma_i_c
            return gamma_i_c
    
    def _M_step2(self, x):
        means_hat = []
        sigmas_hat = []
        gamma_i_c = self.gamma
        gamma_summs = np.sum(gamma_i_c, axis=0)

        v = x[:, np.newaxis, :] - self.means
        v2 =  v[:, :, :, np.newaxis] * v[:, :, np.newaxis, :] * self.pi_c.reshape(-1, 1, 1)
        v_c = np.sum(v2, axis=0)/gamma_summs
        print(f'mean shapes. x: {x.shape} gamma: {gamma_i_c.shape}')
        s_c = np.sum(x[:, :, np.newaxis] * gamma_i_c[:, np.newaxis, :], axis=0)/gamma_summs
        print(f'mean shapes. men: {s_c.shape}')

        pi_hat = gamma_summs/x.shape[0]
        print(f'Before: {self.means.shape}')
        self.means = s_c
        print(f'After: {self.means.shape}')
        self.sigmas = v_c
        self.pi_c = pi_hat
        return means_hat, sigmas_hat, pi_hat
    
    def _M_step(self, x):
        means_hat = []
        sigmas_hat = []
        gamma_i_c = self.gamma
        gamma_summs = np.sum(gamma_i_c, axis=0)
        for c in range(self.n_components):
            s_c = 0
            v_c = 0
            for i in range(x.shape[0]):
                s_c += x[i]*gamma_i_c[i, c]
                v_c += np.outer((x[i]-self.means[c]), (x[i]-self.means[c]))*gamma_i_c[i, c]

            means_hat.append(s_c/gamma_summs[c])
            sigmas_hat.append(v_c/gamma_summs[c])
            pi_hat = gamma_summs/x.shape[0]
        self.means = np.array(means_hat)
        self.sigmas = np.stack(sigmas_hat, axis=0)
        self.pi_c = pi_hat
        return means_hat, sigmas_hat, pi_hat

    def _fit(self, X, y=None):
        _, n_features = X.shape
        self._init_params(self.n_components, n_features)
        for i in range(20):
            print(f'i: {i}')
            print(self.means.shape)
            gamma = self._E_step(X)
            means_hat, sigmas_hat, pi_hat = self._M_step(X)
            print(f'means hat: {means_hat}')
    def _predict(self):
        pass


if __name__ == "__main__":
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(42)

    # Mean and covariance matrix for the first Gaussian
    mean1 = np.array([2, 3])
    cov1 = np.array([[0.1, 0.0], [0.0, 0.1]])

    # Mean and covariance matrix for the second Gaussian
    mean2 = np.array([-10, -20])
    cov2 = np.array([[3, 0], [0, 3]])

    # Number of samples
    num_samples = 20

    # Generate samples from the first Gaussian
    samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)

    # Generate samples from the second Gaussian
    samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)

    # Stack the samples into a single array
    x = np.vstack([samples1, samples2])


    mu = np.array([0, 0])
    sigma = np.eye(2)
    x_0 = np.array([0, 0])
    p = GMM.pdf_multivariate(x_0, mu, sigma)
    print(f'PDF: {p}')

    gmm = GMM(2)
    gmm.fit(x)
    print('+++++++++++++')
    print(gmm.means)
