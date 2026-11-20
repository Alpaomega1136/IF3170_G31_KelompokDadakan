"""Binary kernel SVM trained with a NumPy-only dual working-set optimizer.

The pair selection follows the second-order working-set idea of Fan, Chen and
Lin (2005), JMLR 6:1889-1918. Kernel rows are computed lazily and cached;
neither a library SVM solver nor a full Gram matrix is required.
"""
from collections import OrderedDict
import pickle
import warnings

import numpy as np


class WorkingSetSVM:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3,
                 coef0=1.0, tol=1e-3, max_iter=200000, cache_mb=256,
                 threshold=0.0):
        if C <= 0 or tol <= 0 or max_iter < 1 or cache_mb <= 0:
            raise ValueError("C, tol, max_iter and cache_mb must be positive")
        if kernel not in ("linear", "rbf", "polynomial"):
            raise ValueError("Unknown kernel")
        if gamma not in ("scale", "auto") and (not isinstance(gamma, (int, float)) or gamma <= 0):
            raise ValueError("gamma must be positive, 'scale', or 'auto'")
        if degree < 1:
            raise ValueError("degree must be positive")
        self.C, self.kernel, self.gamma = float(C), kernel, gamma
        self.degree, self.coef0 = int(degree), float(coef0)
        self.tol, self.max_iter = float(tol), int(max_iter)
        self.cache_mb, self.threshold = float(cache_mb), float(threshold)
        self.is_fitted_ = False

    def _kernel(self, a, b):
        product = a @ b.T
        if self.kernel == "linear":
            return product
        if self.kernel == "polynomial":
            return (self.gamma_ * product + self.coef0) ** self.degree
        distance = np.maximum(np.sum(a*a, axis=1)[:, None] +
                              np.sum(b*b, axis=1)[None, :] - 2 * product, 0)
        return np.exp(-self.gamma_ * distance)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y)
        if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite numeric matrix")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per row")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Exactly two classes are required")
        self.n_features_in_ = X.shape[1]
        variance = X.var()
        self.gamma_ = (1.0 / X.shape[1] if self.gamma == "auto" else
                       1.0 / (X.shape[1] * variance) if self.gamma == "scale" and variance else
                       1.0 / X.shape[1] if self.gamma == "scale" else float(self.gamma))
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)
        n = len(y)
        alpha = np.zeros(n)
        gradient = -np.ones(n)
        diagonal = (np.sum(X*X, axis=1) if self.kernel == "linear" else
                    (self.gamma_ * np.sum(X*X, axis=1) + self.coef0) ** self.degree
                    if self.kernel == "polynomial" else np.ones(n))
        row_cache = OrderedDict()
        row_limit = max(1, int(self.cache_mb * 1024**2 // (8 * n)))

        def row(i):
            i = int(i)
            if i in row_cache:
                row_cache.move_to_end(i)
                return row_cache[i]
            values = self._kernel(X[i:i+1], X).reshape(-1)
            row_cache[i] = values
            if len(row_cache) > row_limit:
                row_cache.popitem(last=False)
            return values

        eps = 1e-12
        self.converged_ = False
        for iteration in range(self.max_iter):
            up = ((y_signed > 0) & (alpha < self.C - eps)) | ((y_signed < 0) & (alpha > eps))
            low = ((y_signed > 0) & (alpha > eps)) | ((y_signed < 0) & (alpha < self.C - eps))
            values = -y_signed * gradient
            i = int(np.argmax(np.where(up, values, -np.inf)))
            gap_vector = values[i] - values
            self.kkt_gap_ = float(np.max(np.where(low, gap_vector, -np.inf)))
            if self.kkt_gap_ < self.tol:
                self.converged_ = True
                break
            ki = row(i)
            curvature = np.maximum(diagonal[i] + diagonal - 2 * ki, 1e-12)
            gain = np.where(low & (gap_vector > 0), gap_vector ** 2 / curvature, -np.inf)
            j = int(np.argmax(gain))
            if not np.isfinite(gain[j]):
                break
            limit_i = self.C - alpha[i] if y_signed[i] > 0 else alpha[i]
            limit_j = alpha[j] if y_signed[j] > 0 else self.C - alpha[j]
            step = min(gap_vector[j] / curvature[j], limit_i, limit_j)
            alpha[i] += y_signed[i] * step
            alpha[j] -= y_signed[j] * step
            alpha[i] = np.clip(alpha[i], 0, self.C)
            alpha[j] = np.clip(alpha[j], 0, self.C)
            gradient += step * y_signed * (ki - row(j))
        self.n_iter_ = iteration + 1
        support = alpha > 1e-8
        self.support_vectors_ = X[support].copy()
        self.dual_coef_ = (alpha[support] * y_signed[support]).copy()
        self.n_support_ = int(support.sum())
        free = (alpha > 1e-8) & (alpha < self.C - 1e-8)
        values = -y_signed * gradient
        if free.any():
            self.intercept_ = float(np.mean(values[free]))
        else:
            up = ((y_signed > 0) & (alpha < self.C-eps)) | ((y_signed < 0) & (alpha > eps))
            low = ((y_signed > 0) & (alpha > eps)) | ((y_signed < 0) & (alpha < self.C-eps))
            self.intercept_ = float((np.max(values[up]) + np.min(values[low])) / 2)
        self.is_fitted_ = True
        if not self.converged_:
            warnings.warn(f"SMO reached its limit; KKT gap={self.kkt_gap_:.6g}", RuntimeWarning)
        return self

    def decision_function(self, X, batch_size=256):
        if not self.is_fitted_:
            raise ValueError("Fit the model before prediction")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X must be finite and have the fitted feature count")
        return np.concatenate([self._kernel(X[i:i+batch_size], self.support_vectors_) @ self.dual_coef_
                               + self.intercept_ for i in range(0, len(X), batch_size)])

    def predict(self, X):
        return np.where(self.decision_function(X) >= self.threshold, self.classes_[1], self.classes_[0])

    def save(self, path):
        if not self.is_fitted_:
            raise ValueError("Fit the model before saving")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
