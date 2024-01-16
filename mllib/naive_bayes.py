import numpy as np
from mllib.base import BaseEstimator


class MultinomialNB(BaseEstimator):
    """Multinomial Naive Bayes Classifier.

    Parameters:
    - alpha (float, optional): Smoothing parameter for Laplace smoothing. Default is 0.001.
    """

    def __init__(self, alpha: float = 0.001):
        """
        Initialize the Multinomial Naive Bayes classifier.

        Args:
        - alpha (float, optional): Smoothing parameter for Laplace smoothing. Default is 0.001.
        """
        self.alpha = alpha
        super().__init__()

    def _one_hot_target(self, Y: np.ndarray) -> None:
        """One-hot encode the target variable.

        Args:
        - Y (numpy.ndarray): Target variable.
        """
        unique_vals = np.unique(Y)
        n_targets = len(Y)
        one_hot_targets = np.zeros((n_targets, len(unique_vals)))
        for i, val in enumerate(unique_vals):
            one_hot_targets[:, i] = np.where(Y == val, 1, 0)
            self.original_labels = unique_vals
        self.ohe_targets = one_hot_targets

    def _get_priors(self) -> None:
        """Calculate class priors."""
        self.priors = np.mean(self.ohe_targets, axis=0)

    def get_word_class_probs(self, X: np.ndarray) -> None:
        """Calculate word-class probabilities.

        Args:
        - X (numpy.ndarray): Input feature matrix.
        """
        w_ci = self.ohe_targets.T @ X
        w_ci += self.alpha * np.ones_like(w_ci)
        self.w_ci = w_ci / (np.sum(w_ci, axis=1).reshape(-1, 1) + X.shape[1] + 1)

    def _get_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Calculate log-likelihoods for each class.

        Args:
        - X (numpy.ndarray): Input feature matrix.

        Returns:
        - numpy.ndarray: Log-likelihoods for each class.
        """
        log_priors = np.log(self.priors)
        log_likelihoods = log_priors + X @ np.log(self.w_ci.T)
        return log_likelihoods

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data.

        Args:
        - X (numpy.ndarray): Input feature matrix.
        - y (numpy.ndarray): Target variable.
        """
        self._one_hot_target(y)
        self._get_priors()
        self.get_word_class_probs(X)
        self.fit_required = False

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
        - X (numpy.ndarray): Input feature matrix.

        Returns:
        - numpy.ndarray: Predicted labels.
        """
        likelihoods = self._get_likelihoods(X)
        classes = np.argmax(likelihoods, axis=1)
        original_labels = self.original_labels[classes]
        return original_labels



if __name__ == '__main__':
    import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.naive_bayes import MultinomialNB as skMulltinomialNB

    print('TEST:')
    print('Custom Realization.')
    df = pd.read_csv('assets/data/emails.csv')
    targets = df['Prediction']
    X = df.iloc[:, 1:-1]
    clf = MultinomialNB()
    clf.fit(X, targets)
    pred = clf.predict(X)
    print(classification_report(targets, pred))

    print('\nSclearn Realization:')
    clf2 = skMulltinomialNB()
    clf2.fit(X, targets)
    pred2 = clf2.predict(X)
    print(classification_report(targets, pred2))