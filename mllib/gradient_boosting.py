from base import BaseEstimator
from decision_tree import DecisionTreeRegressor
import numpy as np
from scipy.special import expit


class Loss:
    def loss_fn(self, y, predictions):
        raise NotImplementedError("Subclasses must implement loss_fn method")

    def gradient(self, y, predictions):
        raise NotImplementedError("Subclasses must implement gradient method.")

    def negative_gradient(self, y, predictions):
        return -self.gradient(y, predictions)

    def hessian(self, y, predictions):
        raise NotImplementedError("Subclasses must implement hessian method.")


class RegressionLoss(Loss):
    # MSE Loss
    def loss_fn(self, y, predictions):
        return np.sum((y - predictions)**2)

    def gradient(self, y, predictions):
        return -(y - predictions)

    def hessian(self, y, predictions):
        return np.ones_like(y)


class LogisticLoss(Loss):
    # Logistic Loss
    # L(y, z) = -sum_k_{1 to C} y_k*log(s(z))
    # dL(y, z)/dz_j = s(z_j) - y_j
    def loss_fn(self, y, predictions):
        s = expit(predictions)
        return -np.sum(y * np.log(s))

    def gradient(self, y, predictions):
        s = expit(predictions)
        return s - y

    def hessian(self, y, predictions):
        s = expit(predictions)
        return s * (1 - s)


class BaseBoosting(BaseEstimator):
    def __init__(
            self,
            loss,
            criterion="mse",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=2,
            max_features="div3",
            min_samples_split=2,
    ):
        self.loss = loss
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samplees_split = min_samples_split
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimator_weights = []
        self.estimators = [
            DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samplees_split
            ) for _ in range(self.n_estimators)
        ]
        self.inital_estimator = None

    def _initial_prediction(self, y):
        raise NotImplementedError('Method not implemented')

    def _fit(self, X, y):
        self.inital_estimator = self._initial_prediction(y)
        y_pred = self.inital_estimator*np.ones_like(y)
        for i, estimator in enumerate(self.estimators):
            residuals = self.loss.negative_gradient(y, y_pred)
            estimator.fit(X, residuals)
            predictions = estimator.predict(X)
            w = 1.0 # For future: can optimize w
            self.estimator_weights.append(w)
            y_pred += self.learning_rate * w * predictions
            self.fit_required = False

    def _predict(self, X):
        raise NotImplementedError('Method not implemented')


class GradientBoostingClassifier(BaseBoosting):
    def __init__(
            self,
            loss=LogisticLoss(),
            criterion="mse",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=2,
            max_features="div3",
            min_samples_split=2,
    ):
        super().__init__(
            loss=loss,
            criterion=criterion,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split
        )

    def _initial_prediction(self, y):
        return np.log(np.mean(y == 1) / np.mean(y == 0))

    def predict_proba(self, X):
        X = self._check_x(X)
        if self.fit_required:
            raise ValueError("Fit method should be called first.")
        return self._predict_proba(X)
    
    def _predict_proba(self, X):
        prediction = self.inital_estimator*np.ones(X.shape[0])
        for w, estimator in zip(self.estimator_weights, self.estimators):
            prediction += self.learning_rate * w * estimator.predict(X)
        return expit(prediction)

    def _predict(self, X):
        proba = self._predict_proba(X)
        prediction = np.where(proba > 0.5, 1, 0)
        return prediction


class GradientBoostingRegressor(BaseBoosting):
    def __init__(
            self,
            loss=RegressionLoss(),
            criterion="mse",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=2,
            max_features="div3",
            min_samples_split=2,
    ):
        super().__init__(
            loss=loss,
            criterion=criterion,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split
        )

    def _initial_prediction(self, y):
        return np.mean(y)

    def _predict(self, X):
        prediction = self.inital_estimator*np.ones(X.shape[0])
        for w, estimator in zip(self.estimator_weights, self.estimators):
            prediction += self.learning_rate * w * estimator.predict(X)
        return prediction


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from random_forest import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)

    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='binary'))

    mse = metrics.mean_squared_error
    X, y = make_regression(
        n_features=25, n_informative=15)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.01
    )

    rf = RandomForestRegressor()
    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    pred_rf = rf.predict(X_test)
    print(f"y shape: {y_test.shape} "
          f"gb shape: {pred_gb.shape} rf shape: {pred_rf.shape}")
    print((100*(pred_gb-y_test)/y_test).astype(int))
    print((100*(pred_rf-y_test)/y_test).astype(int))
