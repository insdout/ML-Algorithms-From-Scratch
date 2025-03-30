import numpy as np
from typing import Tuple
from numpy import ndarray
from base import BaseEstimator
from optimizers import SGD


# TODO: Add Kernels

class SVM(BaseEstimator):
    def __init__(self, kernel='linear', C=1.0, tol=1e-2, max_iter=1000, learning_rate=1e-3) -> None:
        super().__init__()
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.kernel = self._get_kernel(kernel)
        self.parameters = None
    
    def gradient_fn(self, X, y, parameters):
        margin = 1 - y * (X @ parameters)
        gradient = np.zeros_like(parameters)
        mask = margin > 0
        gradient = parameters - self.C * np.sum(np.where(mask[:, None], y[:, None] * X, 0), axis=0)
        return gradient
    
    def loss_fn(self, X, y, parameters):
        return np.linalg.norm(parameters[:-1]) / 2 + self.C * np.sum(np.maximum(0, 1 - y * (X @ parameters)))
    
    def _get_kernel(self, kernel_name):
        if kernel_name == 'linear':
            return None
        elif kernel_name == 'rbf':
            return None
        else:
            raise NotImplementedError(f'{kernel_name} not implemented!')
        
    def _check_x(self, X: ndarray) -> ndarray:
        X = super()._check_x(X)
        return np.column_stack((X, np.ones(X.shape[0])))
    
    def _check_input(self, X: ndarray, y: ndarray | None = None) -> Tuple[ndarray, ndarray]:
        X, y = super()._check_input(X, y)
        if set(y) == {0, 1}:
            y = 2 * y - 1
        elif set(y) == {-1, 1}:
            pass
        else:
            raise ValueError("y should be {0, 1} or {-1, 1}")
        return X, y
        
    def _fit(self, X, y):
        self.parameters = np.random.randn(X.shape[1])
        sgd = SGD(
            self.gradient_fn,
            self.parameters,
            self.learning_rate,
            self.loss_fn,
            batch_size=10,
            max_iter=self.max_iter,
            tolerance=self.tol,
            verbose=False
        )
        self.parameters = sgd.optimize(X, y)
        self.hisory = sgd.history
        self.fit_required = False

    def _predict(self, X):
        return np.round((np.sign(X @ self.parameters) + 1) / 2).astype(int)


    
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the SVM model
    svm = SVM(kernel='linear', C=0.001, tol=1e-6, max_iter=1000, learning_rate=1e-2)
    svm.fit(X_train, y_train)

    # Make predictions on the test set using SVM
    y_pred_svm = svm.predict(X_test)
    print(set(y_pred_svm))
    # Initialize and fit the logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Make predictions on the test set using logistic regression
    y_pred_log_reg = log_reg.predict(X_test)

    # Calculate accuracy for SVM
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("Accuracy for SVM:", accuracy_svm)

    # Calculate accuracy for logistic regression
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print("Accuracy for Logistic Regression:", accuracy_log_reg)
