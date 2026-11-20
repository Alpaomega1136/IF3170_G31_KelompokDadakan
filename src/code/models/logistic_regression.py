import pickle
import numpy as np


class LogisticRegression:
    def __init__(self, penalty=None, lam=0.01, l1_ratio=0.5, lr=0.1,
                 n_epoch=50, batch_size=1, tol=1e-6, random_state=42):
        if penalty not in (None, 'l1', 'l2', 'elasticnet'):
            raise ValueError("penalty must be None, 'l1', 'l2', or 'elasticnet'")
        self.penalty, self.lam, self.l1_ratio = penalty, lam, l1_ratio
        self.lr, self.n_epoch, self.batch_size = lr, n_epoch, batch_size
        self.tol, self.random_state = tol, random_state
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _add_bias(X):
        return np.c_[np.ones(len(X)), X]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2D and not empty and y must match the number of rows in X")
        if not np.isfinite(X).all():
            raise ValueError("X cannot be NaN or infinity")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must be binary (0 or 1)")

        X = self._add_bias(X)
        n = len(X)
        self.b = np.zeros(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        # Regularisation: Zou & Hastie (2005); soft-thresholding: Parikh & Boyd (2014); Friedman, Hastie & Tibshirani (2010)
        share_l1 = {None: 0.0, 'l1': 1.0, 'l2': 0.0, 'elasticnet': self.l1_ratio}[self.penalty]
        lam = 0.0 if self.penalty is None else self.lam
        l1, l2 = lam * share_l1, lam * (1 - share_l1)

        self.loss_history_ = []
        for _ in range(self.n_epoch):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                p = self._sigmoid(X[idx] @ self.b)
                step = X[idx].T @ (y[idx] - p) / len(idx)
                step[1:] -= l2 * self.b[1:]
                self.b += self.lr * step
                if l1 > 0:
                    self.b[1:] = np.sign(self.b[1:]) * np.maximum(np.abs(self.b[1:]) - self.lr * l1, 0.0)

            p_all = np.clip(self._sigmoid(X @ self.b), 1e-15, 1 - 1e-15)
            self.loss_history_.append(-np.mean(y * np.log(p_all) + (1 - y) * np.log(1 - p_all)))
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                break
        return self

    def predict_proba(self, X):
        if not hasattr(self, 'b'):
            raise ValueError("Fit the model before prediction")
        return self._sigmoid(self._add_bias(np.asarray(X, float)) @ self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        if not hasattr(self, 'b'):
            raise ValueError("Fit the model before saving")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)