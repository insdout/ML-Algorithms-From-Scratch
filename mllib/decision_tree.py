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
    

def mse_impurity(y):
    y_mean = np.mean(y)
    return np.mean((y - y_mean)**2)

def mae_impurity(y):
    y_mean = np.median(y)
    return np.mean(y - y_mean)


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=3, max_features=1, max_samples_split=2, regression=False):
        self.root = None
        self.criterion = criterion
        if self.criterion == "gini":
            self.criterion_fn = gini_inpurity
        elif self.criterion == "entropy":
            self.criterion_fn = entropy_inpurity
        elif self.criterion == "mse":
            self.criterion_fn = mse_impurity
        else:
            raise ValueError(f"{criterion} is not implemented.")
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples_split = max_samples_split
        self.regression = regression
        if self.regression:
            if self.criterion not in {"mae", "mse"}:
                raise ValueError("For regression criterion should be mae or mse.")
    
    def _check_X(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if X.size == 0:
            raise ValueError("The array X must be non-empty")
        return X
        

    def _check_inputs(self, X, y):
        X = self._check_X(X)

        if y is None:
            raise ValueError("Argument y is required.")
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.size == 0:
            raise ValueError("The array y must be non-empty")
        return X, y 
    
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
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        return (n_left/n_total)*self.criterion_fn(y_left) \
            + (n_right/n_total)*self.criterion_fn(y_right)

    def _best_split(self, X, y, max_features):
        #print("In _best_split!!")
        #print("===============")
        n_features = X.shape[1]

        if max_features is None:
            max_features = X.shape[1]

        #print(f"n_features: {n_features} max_features: {max_features}")
        features_idx = np.random.choice(n_features, max_features, replace=False)

        best_feature_idx = None
        best_threshold = None
        min_impurity = float("inf")

        #print("features_idx", features_idx)
        for feature_idx in features_idx:
            #print(f"feature_idx: {feature_idx}")
            thresholds = self._get_feature_thresholds(X, feature_idx)
            #print(f"thesholds: {thresholds}")
            for threshold in thresholds:
                split_impurity = self._leafs_impurity(X, y, feature_idx, threshold)
                #print(f"feature_idx: {feature_idx} threshold: {threshold} impur: {split_impurity}")
                if split_impurity < min_impurity:
                    #print("Split updated!!!!!!")
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    min_impurity = split_impurity
        return best_feature_idx, best_threshold

    def _is_finished(self, y, depth):
        unique_classes = len(np.unique(y))
        #print(f"depth: {depth} unique: {unique_classes} y size: {y.size}")
        n_samples = y.size
        if (depth > self.max_depth) or (n_samples < self.max_samples_split) or (unique_classes == 1):
            return True
        return False
    
    def _build_tree(self, X, y, depth):
        if self._is_finished(y, depth):
            if self.regression:
                if self.criterion == "mse":
                    prediction = np.mean(y)
                elif self.criterion == "mae":
                    prediction = np.median(y)
                else:
                    raise ValueError(f"{self.criterion} is not allowed for regression! Please use mse or mae.")
            else:
                prediction = np.argmax(np.bincount(y))
            return Node(value=prediction)
        
        feature_idx, threshold = self._best_split(X, y, self.max_features)
        #print(f"depth: {depth} feature_idx: {feature_idx} threshold: {threshold}")
        left_idx, right_idx = self._split_mask(X, feature_idx, threshold)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        print(f"depth: {depth} left child: {left_child} right child: {right_child}")
        return Node(feature_idx=feature_idx, threshold=threshold, left=left_child, right=right_child)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        feature_idx = node.feature_idx
        threshold = node.threshold
        print(f"Node val: {node.value} fet_idx: {node.feature_idx} threshold: {node.threshold} left: {node.left} right: {node.right}")
        print("++++++++++++++++++++++++++++++++")
        if x[feature_idx] <= threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def fit(self, X, y):
        X, y = self._check_inputs(X, y)
        print(f"Fit X type {type(X)}")
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = self._check_X(X)
        predictions = [ self._traverse_tree(x, self.root) for x in X ]
        return np.array(predictions)
 


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion="gini", max_depth=3, max_features=None, max_samples_split=2):
        super().__init__(criterion, max_depth, max_features, max_samples_split, regression=False)
    

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion="mse", max_depth=3, max_features=None, max_samples_split=2):
        super().__init__(criterion, max_depth, max_features, max_samples_split, regression=True)


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
print(f"Outside: X type {type(X_train)}")
tree.fit(X_train, y_train)
print("DT Classifier fitted!")
tree.predict(X_train)
#draw_decision_tree(tree.tree)


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
print(f"Outside: X type {type(X)}")
clf.fit(X, y)