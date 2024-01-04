from base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class CovergenceError(Exception):
    pass


class GMM(BaseEstimator):
    """
    Gaussian Mixture Model (GMM) implementation using Expectation-Maximization (EM) algorithm.

    Parameters:
    - n_components (int): Number of Gaussian components.
    - max_iter (int): Maximum number of iterations for the EM algorithm.
    - tolerance (float): Convergence tolerance.

    Attributes:
    - means (list): List of means for each Gaussian component.
    - sigmas (list): List of covariance matrices for each Gaussian component.
    - likelihood (list): List of likelihood values during training.
    - alpha_c (numpy.ndarray): Prior probabilities for each Gaussian component.
    - gamma (numpy.ndarray): Responsibilities matrix.
    - fit_required (bool): Flag indicating whether the model needs to be fitted.
    """

    y_required = False

    def __init__(self, n_components: int, max_iter: int = 1000, tolerance: float = 1e-6) -> None:
        """
        Initialize the Gaussian Mixture Model.

        Args:
        - n_components (int): Number of Gaussian components.
        - max_iter (int): Maximum number of iterations for the EM algorithm.
        - tolerance (float): Convergence tolerance.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.means = []
        self.sigmas = []
        self.likelihood = []
        self.alpha_c = np.ones(n_components)/n_components

    @staticmethod
    def pdf_multivariate(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Multivariate normal probability density function.

        Args:
        - x (numpy.ndarray): Input data.
        - mu (numpy.ndarray): Mean vector.
        - sigma (numpy.ndarray): Covariance matrix.

        Returns:
        - numpy.ndarray: Probability density values.
        """
        d = len(mu)
        denominator = ((2 * np.pi) ** (d / 2)) * (np.linalg.det(sigma) ** (1 / 2))
        exponent = -0.5 * np.sum(np.dot(x - mu, np.linalg.inv(sigma)) * (x - mu), axis=-1)
        numerator = np.exp(exponent)
        pdf_values = numerator / denominator
        return pdf_values

    def _init_params(self, X: np.ndarray, n_components: int, n_features: int) -> None:
        """
        Initialize the parameters of the Gaussian Mixture Model.

        Args:
        - n_components (int): Number of Gaussian components.
        - n_features (int): Number of features.
        - X (np.ndarray): Input data for initializing means.
        """
        self.fit_required = True
        self.means = np.array([X[ind] for ind in np.random.choice(range(X.shape[0]), self.n_components)])
        self.sigmas = np.stack([np.cov(X.T) for _ in range(self.n_components)])
        self.likelihood = []
        self.alpha_c = np.ones(n_components)/n_components

    def _get_likelihoods(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the likelihoods for each data point and Gaussian component.

        Args:
        - x (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Likelihood values.
        """
        likelihoods = []
        for c in range(self.n_components):
            likelihoods.append(self.pdf_multivariate(x, mu=self.means[c], sigma=self.sigmas[c]))
        return np.stack(likelihoods, axis=1)

    def _E_step(self, x: np.ndarray) -> None:
        """
        Perform the Expectation (E) step of the EM algorithm.

        Args:
        - x (numpy.ndarray): Input data.
        """
        likelihoods = self._get_likelihoods(x)
        self.likelihood.append(likelihoods.sum())
        weighted_likelihoods = likelihoods*self.alpha_c
        self.gamma = weighted_likelihoods/np.sum(weighted_likelihoods, axis=1)[:, np.newaxis]

    def _M_step(self, x: np.ndarray) -> None:
        """
        Perform the Maximization (M) step of the EM algorithm.

        Args:
        - x (numpy.ndarray): Input data.
        """
        gamma_summs = np.sum(self.gamma, axis=0)
        for c in range(self.n_components):
            self.means[c] = np.sum(x*self.gamma[:, c, np.newaxis], axis=0)/gamma_summs[c]
            self.sigmas[c] = ((x - self.means[c]).T @ ((x - self.means[c])*self.gamma[:, c, np.newaxis]))/gamma_summs[c]
        alpha_c = gamma_summs/x.shape[0]
        self.alpha_c = alpha_c

    def _is_converged(self) -> bool:
        """
        Check for convergence based on the change in likelihood.

        Returns:
        - bool: True if converged, False otherwise.
        """
        delta = float('inf')
        if len(self.likelihood) > 1:
            delta = abs(self.likelihood[-2] - self.likelihood[-1])
            print(f'Likelihood delta: {delta:>10.5f} iter: {len(self.likelihood):>3}')
        return delta < self.tolerance

    def _fit(self, X: np.ndarray, y=None) -> None:
        """
        Fit the Gaussian Mixture Model to the input data using the EM algorithm.

        Args:
        - X (numpy.ndarray): Input data.
        - y: Ignored (no supervision required).
        """
        _, n_features = X.shape
        self._init_params(X, self.n_components, n_features)
        for _ in range(self.max_iter):
            self._E_step(X)
            self._M_step(X)
            if self._is_converged():
                self.fit_required = False
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely Gaussian component for each data point.

        Args:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Predicted labels.
        """
        likelihoods = self._get_likelihoods(X)
        preds = np.argmax(likelihoods, axis=1)
        return preds

    def plot(self, X: np.ndarray) -> None:
        """
        Plot the Gaussian Mixture Model on a 3D surface.

        Parameters:
        - X (np.ndarray): Input data for plotting.

        Returns:
        None
        """
        _, n_features = X.shape
        grid_steps = 40
        min_lims = 1.5*np.min(X, axis=0)
        max_lims = 1.5*np.max(X, axis=0)
        xs = np.linspace(min_lims[0], max_lims[0], grid_steps)
        ys = np.linspace(min_lims[1], max_lims[1], grid_steps)
        xv, yv = np.meshgrid(xs, ys)
        x_in = np.c_[xv.ravel(), yv.ravel()]
        zs = np.sum(self._get_likelihoods(x_in), axis=1).reshape(len(xs), len(ys))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xv, yv, zs, cmap='viridis', alpha=1, edgecolors='black', linewidth=0.5)
        plt.savefig('gmm.png')

    def _fit_meshgrid(self, X: np.ndarray) -> None:
        """
        Fit the Gaussian Mixture Model to a 2D meshgrid for visualization.

        Args:
        - X (numpy.ndarray): Input data (ignored in this method).
        """
        _, n_features = X.shape
        self._init_params(X, self.n_components, n_features)
        grid_steps = 40
        min_lims = 1.5*np.min(X, axis=0)
        max_lims = 1.5*np.max(X, axis=0)
        xs = np.linspace(min_lims[0], max_lims[0], grid_steps)
        ys = np.linspace(min_lims[1], max_lims[1], grid_steps)
        xv, yv = np.meshgrid(xs, ys)
        x_in = np.c_[xv.ravel(), yv.ravel()]
        zs = []
        zs.append(np.sum(self._get_likelihoods(x_in), axis=1))
        for i in range(self.max_iter):
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
        """
        Create and save a 3D animation of the fitted Gaussian Mixture Model.

        Args:
        - xv (numpy.ndarray): X values of the meshgrid.
        - yv (numpy.ndarray): Y values of the meshgrid.
        - zs (numpy.ndarray): Likelihood values for each point in the meshgrid.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            ax.plot_surface(xv, yv, zs[frame], cmap='viridis', alpha=1, edgecolors='black', linewidth=0.5)
            ax.set_title(f'Iteration {frame}')
            return []

        ani = animation.FuncAnimation(fig, update, frames=zs.shape[0], blit=True)
        ani.save('gmm_animation.gif', writer='imagemagick', fps=4)


if __name__ == "__main__":
    import numpy as np

    # Set random seed for reproducibility
    #np.random.seed(42)

    # Mean and covariance matrix for the first Gaussian
    mean1 = np.array([-2, 2])
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Mean and covariance matrix for the second Gaussian
    mean2 = np.array([2, 2])
    cov2 = np.array([[1.5, 0], [0, 1.5]])

    mean3 = np.array([2, -2])
    cov3 = np.array([[2, 0], [0, 2]])

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
    gmm = GMM(3)
    gmm.fit(x)
    gmm.plot(x)
    print('+++++++++++++')
    print(gmm.means)
    print('+++++++++++++')
    print(gmm.predict(x))
