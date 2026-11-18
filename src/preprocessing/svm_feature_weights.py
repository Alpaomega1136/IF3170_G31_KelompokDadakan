"""Training-only mutual-information feature weights for SVM preprocessing.

Uses scikit-learn's mutual information estimator in preprocessing; SVM fitting
and prediction remain entirely in the project's NumPy scratch implementation.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class MutualInformationWeights(BaseEstimator, TransformerMixin):
    def __init__(self, exponent=0.5, min_score=0.002, random_state=42):
        self.exponent = exponent
        self.min_score = min_score
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Expected a finite two-dimensional feature matrix")
        discrete = np.array([np.unique(X[:, i]).size < 20 for i in range(X.shape[1])])
        encoded = X.copy()
        for i in np.flatnonzero(discrete):
            _, encoded[:, i] = np.unique(X[:, i], return_inverse=True)
        scores = mutual_info_classif(encoded, y, discrete_features=discrete,
                                     n_neighbors=5, random_state=self.random_state)
        self.scores_ = scores
        mean_score = float(scores.mean())
        self.weights_ = ((np.maximum(scores, self.min_score) / mean_score) ** self.exponent
                         if mean_score > 0 else np.ones(X.shape[1]))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Feature matrix does not match fitted weights")
        return X * self.weights_
