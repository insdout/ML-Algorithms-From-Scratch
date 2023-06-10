import numpy as np


def gini_inpurity(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    probabilities = np.bincount(y)/y.shape[0]
    return 1 - np.sum(probabilities**2)


def entropy_inpurity(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    probabilities = np.bincount(y)/y.shape[0]
    return np.sum(-probabilities * np.log2(probabilities, where=(probabilities > 0)))


def leafs_impurity(y, split):
    y_l = y[split]
    y_r = y[~split]
    return criterion(y_l) + criterion(y_r)
    

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        self.value = None
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=3, max_features=1, regression=False):
        self.root = None
        self.criterion = criterion
        if self.criterion == "gini":
            self.criterion_fn = gini_inpurity
        elif self.criterion == "entropy":
            self.criterion_fn = entropy_inpurity
        else:
            raise ValueError(f"{criterion} is not implemented.")
        self.max_depth = max_depth
        self.max_features = max_features
        self.regression = regression
    
    def _check_X(self, X):
        pass

    def _check_inputs(self, X, y):
        pass
    
    def _get_feature_thresholds(self, X, feature):
        unique_values = np.unique(X[:, feature])
        n_unique = len(unique_values)
        thresholds = []
        for i in range(1, n_unique):
            thresholds.append((unique_values[i-1] + unique_values[i])/2)
        return thresholds


    def _split_mask(self, X, feature, threshold):
        left_idx = (X[:, feature] <= threshold)
        right_idx = (X[:, feature] > threshold)
        return left_idx, right_idx
    
    def _split(self, X, y, feature, threshold):
        left_idx, right_idx = self._split_mask(X, feature, threshold) 
        y_left = y[left_idx]
        y_right = y[right_idx]
        return y_left, y_right
    
    def _leafs_impurity(self, X, y, feature, threshold):
        left_idx, right_idx = self._split_mask(X, feature, threshold) 
        y_left = y[left_idx]
        y_right = y[right_idx]
        return self.criterion_fn(y_left) + self.criterion_fn(y_right)

    def _best_split(self, X, max_features):
        n_features = X.shape[1]
        
        if features is None:
            features = range(X.shape[1])
        else:
            features = np.random.choice(n_features, max_features, replace=False)

        best_feature = None
        best_threshold = None
        min_impurity = float("inf")

        for feature in features:
            thresholds = self._get_feature_thresholds(X, feature)
            for threshold in thresholds:
                split_impurity = self._leafs_impurity(X, y, feature, threshold)
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


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


def draw_decision_tree(tree, indent=''):
    if isinstance(tree, dict):
        feature_idx = tree['feature_idx']
        threshold = tree['threshold']
        print(indent + 'X[' + str(feature_idx) + '] <= ' + str(threshold))
        draw_decision_tree(tree['left'], indent + '    /')
        draw_decision_tree(tree['right'], indent + '    \\')
    else:
        print(indent + str(tree))

from sklearn.datasets import make_classification
X_train, y_train = make_classification(
    n_features=6, n_redundant=0, n_informative=4, random_state=1, n_clusters_per_class=1
)
tree = DecisionTreeClassifier(criterion='gini', max_depth=2)
tree.fit(X_train, y_train)
draw_decision_tree(tree.tree)


def height(tree):
    if isinstance(tree, dict):
        left_height = height(tree['left'])
        right_height = height(tree['right'])
        return max(left_height, right_height) + 1
    else:
        return 0


def print_tree(tree):
    h = height(tree)
    col = 2 ** h - 1
    M = [[' ' for _ in range(col * 2 - 1)] for _ in range(h)]
    print_tree_helper(M, tree, col - 1, 0, h)


def print_tree_helper(M, tree, col, row, height):
    if isinstance(tree, dict):
        feature_idx = tree['feature_idx']
        threshold = tree['threshold']
        M[row][int(col)] = f"X[{feature_idx}] <= {threshold}"

        level_widths = [2 ** (h - row - 1) for h in range(height)]
        level_width = sum(level_widths[:height - row - 1])

        if len(M[row + 1]) <= col - level_width:
            M[row + 1].extend([' '] * (col - level_width - len(M[row + 1]) + 1))
        if len(M[row + 1]) <= col + level_width:
            M[row + 1].extend([' '] * (col + level_width - len(M[row + 1]) + 1))

        print_tree_helper(M, tree['left'], col - level_width, row + 1, height)
        print_tree_helper(M, tree['right'], col + level_width, row + 1, height)
    else:
        M[row][int(col)] = str(tree)

# Example usage:
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)