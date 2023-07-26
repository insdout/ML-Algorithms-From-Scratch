import numpy as np
from base import BaseEstimator
from mllib.optimizers import SGD


class LogisticRegression(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, tolerance=1e-2, learning_rate=1e-1):
        self.C = C
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate
    
    def loss_fn(self, X_intercept, y, parameters):
        n = y.shape[0]
        probabilities = self.sigmoid(X_intercept @ parameters)
        return (1/n)*(-self.C*np.sum(y*np.log(probabilities) + (1-y)*np.log(1-probabilities)) \
                       + 0.5*np.sum(np.concatenate(([0], parameters[1:]))**2))

    def gradient_fn(self, X_intercept, y, parameters):
        n = y.shape[0]
        probabilities = self.sigmoid(X_intercept @ parameters)
        #return (1/n)*(np.sum(X_intercept.T @ (probabilities - y)) + np.sum(np.concatenate(([0], parameters[1:]))))
        return (1/n)*(self.C*X_intercept.T @ (probabilities - y) + np.sum(np.concatenate(([0], parameters[1:]))))
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit(self, X, y):
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

    def _predict(self, X):
        return np.round(self.predict_proba(X))
    
    def predict_proba(self, X):
        X = self._check_x(X)
        X_intercept = self._add_intercept(X)
        return self.sigmoid(X_intercept @ self.parameters)



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from  sklearn.linear_model import LogisticRegression as LogisticRegression_gt
    from sklearn import metrics
    X, y = make_classification(
        n_features=6, n_redundant=0, n_informative=4, n_clusters_per_class=1
    )
    lr_gt = LogisticRegression_gt(penalty="l2", C=1)
    lr_gt.fit(X, y)
    print(np.c_[lr_gt.intercept_, lr_gt.coef_])
    pred_gt = lr_gt.predict(X)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y, pred_gt))
    print()

    lr = LogisticRegression(tolerance=1e-5, learning_rate=1e-1, C=1, max_iter=1000)
    lr.fit(X, y)
    print(lr.parameters)
    pred1 = lr.predict(X)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y, pred1))
    print()

    lr2 = LogisticRegression(tolerance=1e-5, learning_rate=1e-1, C=1, max_iter=1000)
    lr2.fit(X, y)
    print(lr2.parameters)
    pred2 = lr2.predict(X)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y, pred2))
  
    