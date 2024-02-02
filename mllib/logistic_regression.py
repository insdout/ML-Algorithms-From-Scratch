import numpy as np
from base import BaseEstimator
from optimizers import SGD


class LogisticRegression(BaseEstimator):
    def __init__(
        self,
        C: float = 1,
        max_iter: int = 1000,
        tolerance: float = 1e-2,
        learning_rate: float = 1e-1
    ) -> None:
        """
        Initialize logistic regression model.

        Parameters:
            - C (float): Regularization parameter (default: 1).
            - max_iter (int): Maximum number of iterations for optimization (default: 1000).
            - tolerance (float): Tolerance for convergence (default: 1e-2).
            - learning_rate (float): Learning rate for stochastic gradient descent (default: 1e-1).
        """
        self.C = C
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def loss_fn(self, X_intercept: np.ndarray, y: np.ndarray, parameters: np.ndarray) -> float:
        """
        Compute logistic regression loss.

        Parameters:
            - X_intercept (np.ndarray): Input matrix with intercept.
            - y (np.ndarray): Target values.
            - parameters (np.ndarray): Model parameters.

        Returns:
            - float: Logistic regression loss.
        """
        n = y.shape[0]
        probabilities = self.sigmoid(X_intercept @ parameters)
        return (1/n)*(-self.C*np.sum(
                y*np.log(probabilities)
                + (1-y)*np.log(1-probabilities)
            )
            + 0.5*np.sum(np.concatenate(([0], parameters[1:]))**2))

    def gradient_fn(self, X_intercept: np.ndarray, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic regression loss.

        Parameters:
            - X_intercept (np.ndarray): Input matrix with intercept.
            - y (np.ndarray): Target values.
            - parameters (np.ndarray): Model parameters.

        Returns:
            - np.ndarray: Gradient of the logistic regression loss.
        """
        n = y.shape[0]
        probabilities = self.sigmoid(X_intercept @ parameters)
        return (1/n)*(self.C*X_intercept.T @ (probabilities - y)
                      + np.sum(np.concatenate(([0], parameters[1:]))))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Parameters:
            - x (np.ndarray): Input array.

        Returns:
            - np.ndarray: Sigmoid values.
        """
        return 1/(1 + np.exp(-x))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add intercept term to the input matrix.

        Parameters:
            - X (np.ndarray): Input matrix.

        Returns:
            - np.ndarray: Input matrix with intercept.
        """
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit logistic regression model.

        Parameters:
            - X (np.ndarray): Input matrix.
            - y (np.ndarray): Target values.
        """
        X, y = self._check_input(X, y)
        X_intercept = self._add_intercept(X)
        n_parameters = X_intercept.shape[1]
        self.parameters = np.random.rand(n_parameters)

        sgd = SGD(
            self.gradient_fn,
            self.parameters,
            self.learning_rate,
            self.loss_fn,
            batch_size=10,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            verbose=False
        )
        self.parameters = sgd.optimize(X_intercept, y)
        self.hisory = sgd.history
        self.fit_required = False

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Parameters:
            - X (np.ndarray): Input matrix.

        Returns:
            - np.ndarray: Predicted binary labels.
        """
        return np.round(self.predict_proba(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters:
            - X (np.ndarray): Input matrix.

        Returns:
            - np.ndarray: Predicted class probabilities.
        """
        X = self._check_x(X)
        X_intercept = self._add_intercept(X)
        return self.sigmoid(X_intercept @ self.parameters)



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn import metrics
    X, y = make_classification(
        n_features=6,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1
    )

    lr = LogisticRegression(
        tolerance=1e-5,
        learning_rate=1e-1,
        C=1,
        max_iter=1000
        )
    lr.fit(X, y)
    print(lr.parameters)
    pred1 = lr.predict(X)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y, pred1))
    print()

    lr2 = LogisticRegression(
        tolerance=1e-5,
        learning_rate=1e-1,
        C=1,
        max_iter=1000
        )
    lr2.fit(X, y)
    print(lr2.parameters)
    pred2 = lr2.predict(X)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y, pred2))
