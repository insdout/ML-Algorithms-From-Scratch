import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, List, Any, Generator

class BaseEstimator:
    y_required: bool = True
    fit_required: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEstimator':
        X, y = self._check_input(X, y)
        self._fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_x(X)
        if self.fit_required:
            raise ValueError("Fit method should be called first.")
        return self._predict(X)

    def _check_x(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.size == 0:
            raise ValueError("The array X must be non-empty")
        elif X.ndim > 2:
            raise ValueError("Input must be a 2-dimensional array.")
        elif X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        return X

    def _check_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X = self._check_x(X)

        if self.y_required:
            if y is None:
                raise ValueError("Argument y is required.")
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if y.size == 0:
                raise ValueError("The array y must be non-empty")
        return X, y

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError("Subclasses must implement _fit method.")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _predict method.")


class BaseOptimizer:
    def __init__(
            self,
            gradient_fn: Any,
            parameters: List[float],
            learning_rate: float,
            batch_size: int,
            tolerance: float = 1e-8,
            max_iter: int = 1000
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_fn = gradient_fn
        self.parameters = parameters
        self.tol = tolerance
        self.max_iter = max_iter
        self.history = defaultdict(list)
        self.history["parameters"].append(list(parameters))

    def batch_generator(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = X.shape[0]
        batch_size = self.batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]

    def clear_history(self) -> None:
        self.history = defaultdict(list)

    def update_parameters(self, parameters: List[float], gradient: np.ndarray) -> None:
        raise NotImplementedError(
            "Subclasses must implement update_parameters method.")

    def optimize(self) -> None:
        raise NotImplementedError("Subclasses must implement optimize method.")

