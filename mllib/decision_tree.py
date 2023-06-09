import numpy as np

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = {}

    def _impurity(self, y):
        if self.criterion == 'gini':
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            impurity = 1 - np.sum(probabilities**2)
        elif self.criterion == 'entropy':
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            impurity = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError("Invalid criterion specified. Supported criteria: 'gini', 'entropy'.")
        return impurity

    def _best_split(self, X, y):
        best_feature_idx = None
        best_threshold = None
        best_gain = -np.inf

        n_features = X.shape[1]
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            for threshold in values:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold

                left_impurity = self._impurity(y[left_indices])
                right_impurity = self._impurity(y[right_indices])

                total_impurity = (len(y[left_indices]) / len(y)) * left_impurity + \
                                 (len(y[right_indices]) / len(y)) * right_impurity

                information_gain = self._impurity(y) - total_impurity

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            leaf_value = np.mean(y)
            return leaf_value

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None or threshold is None:
            leaf_value = np.mean(y)
            return leaf_value

        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold

        tree = {'feature_idx': feature_idx,
                'threshold': threshold,
                'left': self._build_tree(X[left_indices], y[left_indices], depth + 1),
                'right': self._build_tree(X[right_indices], y[right_indices], depth + 1)}

        return tree

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        if isinstance(tree, dict):
            feature_idx = tree['feature_idx']
            threshold = tree['threshold']

            if x[feature_idx] <= threshold:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        else:
            return tree

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._predict_sample(x, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion='gini', max_depth=None):
        super().__init__(criterion, max_depth)

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == 'gini':
            impurity = 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            impurity = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError("Invalid criterion specified. Supported criteria: 'gini', 'entropy'.")
        
        return impurity
    

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion='mse', max_depth=None):
        super().__init__(criterion, max_depth)

    def _impurity(self, y):
        if self.criterion == 'mse':
            impurity = np.mean((y - np.mean(y)) ** 2)
        elif self.criterion == 'mae':
            impurity = np.mean(np.abs(y - np.mean(y)))
        else:
            raise ValueError("Invalid criterion specified. Supported criteria: 'mse', 'mae'.")
        
        return impurity

