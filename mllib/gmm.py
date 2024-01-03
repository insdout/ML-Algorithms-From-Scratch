from base import BaseEstimator
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class CovergenceError(Exception):
    pass


class GMM(BaseEstimator):
    y_required = False

    def __init__(self, n_components: int, max_iter: int = 1000, tolerance: float = 1e-8) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.means = []
        self.sigmas = []
        self.likelihood = []
        self.pi_c = np.ones(n_components)/n_components

    @staticmethod
    def pdf_multivariate(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        d = len(mu)
        denominator = ((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2))
        exponent = -0.5 * np.sum(np.dot(x - mu, np.linalg.inv(sigma)) * (x - mu), axis=-1)
        numerator = np.exp(exponent)
        pdf_values = numerator / denominator
        return pdf_values

    def _init_params(self, n_components: int, n_features: int) -> None:
        self.fit_required = True
        self.means = 2*np.random.rand(n_components, n_features)
        sigmas = [(np.random.rand(1) + 0.2)*np.eye(n_features) for _ in range(n_components)]
        self.sigmas = np.stack(sigmas, axis=0)
        self.likelihood = []
        self.pi_c = np.ones(n_components)/n_components

    def _get_likelihoods(self, x: np.ndarray) -> np.ndarray:
        likelihoods = []
        for c in range(self.n_components):
            likelihoods.append(self.pdf_multivariate(x, mu=self.means[c], sigma=self.sigmas[c]))
            #likelihoods.append(multivariate_normal.pdf(x, self.means[c], self.sigmas[c]))
        return np.stack(likelihoods, axis=1)

    def _E_step(self, x):
        likelihoods = self._get_likelihoods(x)
        self.likelihood.append(likelihoods.sum())
        weighted_likelihoods = likelihoods*self.pi_c
        self.gamma = weighted_likelihoods/np.sum(weighted_likelihoods, axis=1)[:, np.newaxis]

    def _M_step(self, x):
        gamma_summs = np.sum(self.gamma, axis=0)
        for c in range(self.n_components):
            self.means[c] = np.sum(x*self.gamma[:, c, np.newaxis], axis=0)/gamma_summs[c]
            self.sigmas[c] = ((x - self.means[c]).T @ ((x - self.means[c])*self.gamma[:, c, np.newaxis]))/gamma_summs[c]
        pi_c = gamma_summs/x.shape[0]
        self.pi_c = pi_c

    def _is_converged(self):
        delta = float('inf')
        if len(self.likelihood) > 1:
            delta = abs(self.likelihood[-2] - self.likelihood[-1])
            print(f'Likelihood delta: {delta:>10.5f} iter: {len(self.likelihood):>3}')
        return delta < self.tolerance

    def _fit(self, X, y=None):
        _, n_features = X.shape
        self._init_params(self.n_components, n_features)
        for i in range(self.max_iter):
            self._E_step(X)
            self._M_step(X)
            if self._is_converged():
                self.fit_required = False
                break

    def _predict(self, X):
        likelihoods = self._get_likelihoods(X)
        preds = np.argmax(likelihoods, axis=1)
        return preds
    
    def _fit_meshgrid(self, X):
        _, n_features = X.shape
        self._init_params(self.n_components, n_features)
        grid_steps = 40
        min_lims = 1.5*np.min(X, axis=0)
        max_lims = 1.5*np.max(X, axis=0)
        xs = np.linspace(min_lims[0], max_lims[0], grid_steps)
        ys = np.linspace(min_lims[1], max_lims[1], grid_steps)
        xv, yv = np.meshgrid(xs, ys)
        x_in = np.c_[xv.ravel(), yv.ravel()]
        zs = []
        zs.append(np.sum(self._get_likelihoods(x_in), axis=1))
        for i in range(100):
            self._E_step(X)
            self._M_step(X)

            zs.append(np.sum(self._get_likelihoods(x_in), axis=1))
            if self._is_converged():
                self.fit_required = False
                break
        zs_len = len(zs)
        zs = np.array(zs).reshape((zs_len), len(xs), len(ys))
        self.plot_gif(xv, yv, zs)

    def plot_gif(self, xv, yv, zs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            ax.plot_surface(xv, yv, zs[frame], cmap='viridis', alpha=0.9, antialiased=True)
            # Contour
            #ax.contour(xv, yv, zs[frame], cmap='viridis', offset=0.2)

            # Z-axis limits
            #ax.set(zlim = (0, 0.5))
            #ax.plot_wireframe(xv, yv, zs[frame], cmap='summer', alpha=0.9, antialiased=True)
            ax.set_title(f'Iteration {frame}')
            return []

        ani = animation.FuncAnimation(fig, update, frames=zs.shape[0], blit=True)
        ani.save('gmm_animation.gif', writer='imagemagick', fps=4)

    




if __name__ == "__main__":
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(42)

    # Mean and covariance matrix for the first Gaussian
    mean1 = np.array([-2, 2])
    cov1 = np.array([[1, 0.0], [0.0, 1]])

    # Mean and covariance matrix for the second Gaussian
    mean2 = np.array([2, 2])
    cov2 = np.array([[1, 0], [0, 1]])

    mean3 = np.array([2, -2])
    cov3 = np.array([[1, 0], [0, 1]])

    # Number of samples
    num_samples = 50

    # Generate samples from the first Gaussian
    samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)

    # Generate samples from the second Gaussian
    samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)

    samples3 = np.random.multivariate_normal(mean3, cov3, num_samples)

    # Stack the samples into a single array
    x = np.vstack([samples1, samples2, samples3])


    gmm = GMM(3)
    print('X shape', x.shape)
    print(gmm._fit_meshgrid(x))
    gmm.fit(x)
    print('+++++++++++++')
    print(gmm.means)
    print('+++++++++++++')
    print(gmm.predict(x))
    print('+++++++++++++')
    print(gmm._fit_meshgrid(x))
