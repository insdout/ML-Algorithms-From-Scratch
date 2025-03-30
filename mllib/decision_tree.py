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
    def __init__(self, criterion="gini", max_depth=3, max_features=1, min_samples_split=2, regression=False):
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
        self.min_samples_split = min_samples_split
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

    def _best_split(self, X, y):
        n_features = X.shape[1]

        if self.max_features is None:
            max_features = n_features
        elif self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        elif self.max_features == "div3":
            max_features = int(n_features/3)
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            raise ValueError("max_features could be: None, 'sqrt', 'log2' or int.")
        max_features = max(max_features, 1)

        features_idx = np.random.choice(n_features, max_features, replace=False)

        best_feature_idx = None
        best_threshold = None
        min_impurity = float("inf")

        for feature_idx in features_idx:
            thresholds = self._get_feature_thresholds(X, feature_idx)
            for threshold in thresholds:
                split_impurity = self._leafs_impurity(X, y, feature_idx, threshold)
                if split_impurity < min_impurity:
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    min_impurity = split_impurity
        return best_feature_idx, best_threshold

    def _is_finished(self, y, depth):
        unique_classes = len(np.unique(y))
        n_samples = y.size
        if (self.max_depth and (depth >= self.max_depth)) or (n_samples < self.min_samples_split) or (unique_classes == 1):
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
                prediction = np.bincount(y, minlength=self.n_classes_)/y.shape[0]
            return Node(value=prediction)

        feature_idx, threshold = self._best_split(X, y)
        left_idx, right_idx = self._split_mask(X, feature_idx, threshold)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(feature_idx=feature_idx, threshold=threshold, left=left_child, right=right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        feature_idx = node.feature_idx
        threshold = node.threshold
        if x[feature_idx] <= threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        X, y = self._check_inputs(X, y)
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method.")


def print_tree(tree):
    def height(root):
        if root is None:
            return 0
        return max(height(root.left), height(root.right))+1

    def getcol(h):
        if h == 1:
            return 1
        return getcol(h-1) + getcol(h-1) + 1

    def printTree(M, root, col, row, height, type):
        if root is None:
            return
        if root.is_leaf():
            M[row][col] = f"{ root.value}"
        else:
            if type == "left":
                M[row][col] = f"X[{root.feature_idx}]<={root.threshold :02.1f}"
            else:
                M[row][col] = f"X[{root.feature_idx}]> {root.threshold :02.1f}"
        printTree(M, root.left, col-pow(2, height-2), row+1, height-1, "left")
        printTree(M, root.right, col+pow(2, height-2), row+1, height-1, "right")

    h = height(tree.root)
    col = getcol(h)
    M = [[0 for _ in range(col)] for __ in range(h)]
    printTree(M, tree.root, col//2, 0, h, "left")
    for i in M:
        for j in i:
            if j == 0:
                print("", end="")
            else:
                print(j, end="   ")
        print("")


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion="gini", max_depth=3, max_features=None, min_samples_split=2):
        super().__init__(criterion, max_depth, max_features, min_samples_split, regression=False)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        X = self._check_X(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion="mse", max_depth=3, max_features=None, min_samples_split=2):
        super().__init__(criterion, max_depth, max_features, min_samples_split, regression=True)

    def predict(self, X):
        X = self._check_X(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    X_train, y_train = make_classification(
        n_features=6, n_redundant=0, n_informative=4, random_state=1, n_clusters_per_class=1
    )
    tree = DecisionTreeClassifier(criterion='gini', max_depth=2)
    tree.fit(X_train, y_train)
    print("DT Classifier fitted!")
    tree.predict(X_train)
    # draw_decision_tree(tree.tree)
    print_tree(tree)

    X_train, y_train = make_regression(
        n_features=6, n_informative=4, random_state=1)
    tree = DecisionTreeRegressor(criterion='mse', max_depth=2)
    tree.fit(X_train, y_train)
    print("DT Regressor fitted!")
    tree.predict(X_train)
    # draw_decision_tree(tree.tree)
    print_tree(tree)

    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X, y = make_classification(
        n_features=20, n_redundant=2, n_informative=15, random_state=42, n_clusters_per_class=3, class_sep=0.1, n_classes=2
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=10)
    rf.fit(X_train, y_train)
    print("=============")
    print_tree(rf)
    print()
    pred = rf.predict(X_test)
    pred_prob = rf.predict_proba(X_test)
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print(f"predictions: {pred[:10]}")
    print(f"predictions proba: {pred_prob}")
    print("Done")

