import pickle
from numbers import Real

import numpy as np


class DecisionTreeID3:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    @staticmethod
    def _missing(value):
        return value is None or (isinstance(value, Real) and np.isnan(value))

    @staticmethod
    def _entropy(counts):
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=-1, keepdims=True)
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)
        logs = np.zeros_like(probabilities)
        np.log2(probabilities, out=logs, where=probabilities > 0)
        return -(probabilities * logs).sum(axis=-1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D array and y must match its rows")
        if any(self._missing(value) for value in y):
            raise ValueError("y must not contain missing labels")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.numeric_ = []
        for column in X.T:
            values = [value for value in column if not self._missing(value)]
            numeric = all(isinstance(value, Real) for value in values)
            if numeric and any(not np.isfinite(value) for value in values):
                raise ValueError("Numeric features must not contain infinity")
            if not numeric and not all(isinstance(value, str) for value in values):
                raise ValueError("Each feature must contain only numbers or only strings")
            self.numeric_.append(numeric)
        self.tree_ = self._grow(X, encoded, list(range(X.shape[1])), 0)
        return self

    def _grow(self, X, y, features, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = {"prediction": self.classes_[np.argmax(counts)]}
        min_leaf = getattr(self, "min_samples_leaf", 1)
        if (np.count_nonzero(counts) == 1 or len(y) < self.min_samples_split
                or not features or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        parent_entropy = self._entropy(counts)
        best_gain = -1.0
        best_split = None
        for feature in features:
            column = X[:, feature]
            missing = np.fromiter((self._missing(value) for value in column), bool, len(column))
            if missing.any() and missing.sum() < min_leaf:
                continue
            missing_counts = np.bincount(y[missing], minlength=len(self.classes_))
            missing_entropy = missing.sum() * self._entropy(missing_counts)

            if self.numeric_[feature]:
                valid = ~missing
                values = column[valid].astype(float)
                if len(values) == 0:
                    continue
                if missing.any() and len(values) >= min_leaf:
                    valid_counts = np.bincount(y[valid], minlength=len(self.classes_))
                    gain = parent_entropy - (len(values) * self._entropy(valid_counts)
                                             + missing_entropy) / len(y)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature, np.max(values), missing)
                order = np.argsort(values, kind="stable")
                values, labels = values[order], y[valid][order]
                positions = np.flatnonzero(values[:-1] < values[1:])
                positions = positions[(positions + 1 >= min_leaf) & (len(values) - positions - 1 >= min_leaf)]
                if len(positions) == 0:
                    continue
                one_hot = np.eye(len(self.classes_), dtype=int)[labels]
                cumulative = np.cumsum(one_hot, axis=0)
                left = cumulative[positions]
                right = cumulative[-1] - left
                left_size = positions + 1
                right_size = len(values) - left_size
                gain = parent_entropy - (left_size * self._entropy(left)
                    + right_size * self._entropy(right) + missing_entropy) / len(y)
                index = np.argmax(gain)
                if gain[index] > best_gain:
                    threshold = values[positions[index]] / 2 + values[positions[index] + 1] / 2
                    best_gain = gain[index]
                    best_split = (feature, threshold, missing)
            else:
                categories = np.unique(column[~missing])
                if len(categories) + bool(missing.any()) < 2:
                    continue
                if any(np.count_nonzero((~missing) & (column == category)) < min_leaf for category in categories):
                    continue
                weighted = missing_entropy
                for category in categories:
                    labels = y[(~missing) & (column == category)]
                    weighted += len(labels) * self._entropy(np.bincount(labels, minlength=len(self.classes_)))
                gain = parent_entropy - weighted / len(y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, None, missing)

        if best_split is None:
            return node
        feature, threshold, missing = best_split
        node.update({"feature": feature, "threshold": threshold, "children": {}})
        column = X[:, feature]
        next_features = features if self.numeric_[feature] else [f for f in features if f != feature]
        if threshold is None:
            for category in np.unique(column[~missing]):
                mask = (~missing) & (column == category)
                node["children"][category] = self._grow(X[mask], y[mask], next_features, depth + 1)
        else:
            valid_indices = np.flatnonzero(~missing)
            values = column[~missing].astype(float)
            for key, select in [("left", values <= threshold), ("right", values > threshold)]:
                indices = valid_indices[select]
                if len(indices):
                    node["children"][key] = self._grow(X[indices], y[indices], next_features, depth + 1)
        if missing.any():
            node["children"][None] = self._grow(X[missing], y[missing], next_features, depth + 1)
        return node

    def predict(self, X):
        if self.tree_ is None:
            raise ValueError("Fit the model before prediction")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used in training")
        predictions = []
        for row in X:
            node = self.tree_
            while "feature" in node:
                feature = node["feature"]
                value = row[feature]
                if self._missing(value):
                    key = None
                elif node["threshold"] is None:
                    key = value
                else:
                    if not isinstance(value, Real) or not np.isfinite(value):
                        raise ValueError("Numeric features must contain finite numbers or missing values")
                    key = "left" if value <= node["threshold"] else "right"
                if key not in node["children"]:
                    break
                node = node["children"][key]
            predictions.append(node["prediction"])
        return np.asarray(predictions, dtype=self.classes_.dtype)

    def save(self, path):
        if self.tree_ is None:
            raise ValueError("Fit the model before saving")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
