"""Binary kernel SVM trained from scratch with Platt's simplified SMO.

The implementation uses NumPy for kernel and optimization arithmetic. It does
not call a scikit-learn SVM solver. Inputs are expected to be finite numeric
features (typically the output of the project's fitted preprocessor). The
two-variable dual update follows the simplified SMO method described in
Platt, J. C. (1998), "Sequential Minimal Optimization: A Fast Algorithm for
Training Support Vector Machines", Microsoft Research Technical Report
MSR-TR-98-14.
"""

from __future__ import annotations

import pickle
from numbers import Integral, Real
from pathlib import Path

import numpy as np


class KernelSVM:
    """C-SVM classifier for binary labels using Linear, Polynomial, or RBF kernels.

    Parameters
    ----------
    C : float
        Soft-margin penalty; must be positive.
    kernel : {"linear", "polynomial", "rbf", "gaussian"}
        Kernel function. ``gaussian`` is an alias for ``rbf``.
    degree : int
        Polynomial degree (only used by the polynomial kernel).
    gamma : float or {"scale", "auto"}, optional
        Kernel scale. ``scale`` means ``1 / (n_features * X.var())`` and
        ``auto`` means ``1 / n_features``. For RBF/poly, omitted gamma uses
        ``scale``.
    coef0 : float
        Additive constant for the polynomial kernel.
    tol : float
        KKT tolerance used by SMO.
    max_passes : int
        Stop after this many consecutive full passes without an update.
    max_iter : int
        Hard upper bound on the number of training examples examined.
    max_kernel_matrix_bytes : int
        Safety cap for the dense Gram matrix. This keeps quadratic memory use
        explicit rather than risking an uncontrolled allocation.
    random_state : int
        Seed used to choose the initial partner in SMO.
    """

    def __init__(
        self,
        C=1.0,
        kernel="linear",
        degree=3,
        gamma="scale",
        coef0=1.0,
        tol=1e-3,
        max_passes=5,
        max_iter=20000,
        max_kernel_matrix_bytes=512 * 1024 * 1024,
        random_state=42,
    ):
        if not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
            raise ValueError("C must be a finite positive number")
        kernel = str(kernel).lower()
        if kernel == "gaussian":
            kernel = "rbf"
        if kernel not in {"linear", "polynomial", "rbf"}:
            raise ValueError("kernel must be 'linear', 'polynomial', or 'rbf'")
        if not isinstance(degree, Integral) or degree < 1:
            raise ValueError("degree must be a positive integer")
        if gamma not in {"scale", "auto", None} and (
            not isinstance(gamma, Real) or not np.isfinite(gamma) or gamma <= 0
        ):
            raise ValueError("gamma must be positive, 'scale', 'auto', or None")
        if not isinstance(coef0, Real) or not np.isfinite(coef0):
            raise ValueError("coef0 must be finite")
        if not isinstance(tol, Real) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite positive number")
        if not isinstance(max_passes, Integral) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        if not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(max_kernel_matrix_bytes, Integral) or max_kernel_matrix_bytes < 8:
            raise ValueError("max_kernel_matrix_bytes must be at least 8")

        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.max_iter = int(max_iter)
        self.max_kernel_matrix_bytes = int(max_kernel_matrix_bytes)
        self.random_state = int(random_state)
        self.is_fitted_ = False

    @staticmethod
    def _as_2d_finite(X, *, name):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"{name} must be a nonempty 2D numeric array")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    def _resolve_gamma(self, X):
        if isinstance(self.gamma, Real):
            return float(self.gamma)
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        variance = float(X.var())
        return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0 / X.shape[1]

    def _kernel_matrix(self, A, B):
        dot = A @ B.T
        if self.kernel == "linear":
            return dot
        if self.kernel == "polynomial":
            return (self.gamma_ * dot + self.coef0) ** self.degree
        a2 = np.einsum("ij,ij->i", A, A)[:, None]
        b2 = np.einsum("ij,ij->i", B, B)[None, :]
        squared_distance = np.maximum(a2 + b2 - 2.0 * dot, 0.0)
        return np.exp(-self.gamma_ * squared_distance)

    def fit(self, X, y):
        X = self._as_2d_finite(X, name="X")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array with one label per row of X")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("KernelSVM currently supports exactly two classes")
        required_bytes = len(X) * len(X) * np.dtype(np.float64).itemsize
        if required_bytes > self.max_kernel_matrix_bytes:
            raise MemoryError(
                f"Dense kernel matrix requires {required_bytes / 1024**2:.1f} MiB, "
                f"above max_kernel_matrix_bytes={self.max_kernel_matrix_bytes}. "
                "Use a smaller training subset or raise the cap deliberately."
            )

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.gamma_ = self._resolve_gamma(X)
        y_signed = np.where(y == classes[1], 1.0, -1.0)
        gram = self._kernel_matrix(X, X)
        n_samples = len(X)
        alphas = np.zeros(n_samples, dtype=np.float64)
        errors = -y_signed.copy()  # f(x_i) - y_i when alpha and b are zero
        bias = 0.0
        passes_without_change = 0
        examined = 0
        updates = 0
        rng = np.random.default_rng(self.random_state)

        while passes_without_change < self.max_passes and examined < self.max_iter:
            changed = 0
            for i in rng.permutation(n_samples):
                if examined >= self.max_iter:
                    break
                examined += 1
                yi, ai, Ei = y_signed[i], alphas[i], errors[i]
                violates = (yi * Ei < -self.tol and ai < self.C - self.tol) or (
                    yi * Ei > self.tol and ai > self.tol
                )
                if not violates:
                    continue

                # Start with the point whose cached error is farthest away;
                # try a few alternatives when the pair has no feasible gain.
                error_gap = np.abs(Ei - errors)
                error_gap[i] = -np.inf
                candidate_count = min(32, n_samples - 1)
                candidate_pool = np.argpartition(error_gap, -candidate_count)[-candidate_count:]
                candidates = candidate_pool[np.argsort(-error_gap[candidate_pool])]
                updated = False
                for j in candidates[: min(32, n_samples - 1)]:
                    yj, aj, Ej = y_signed[j], alphas[j], errors[j]
                    if yi != yj:
                        low = max(0.0, aj - ai)
                        high = min(self.C, self.C + aj - ai)
                    else:
                        low = max(0.0, ai + aj - self.C)
                        high = min(self.C, ai + aj)
                    if high - low <= 1e-12:
                        continue

                    eta = gram[i, i] + gram[j, j] - 2.0 * gram[i, j]
                    if eta <= 1e-12:
                        continue
                    new_aj = np.clip(aj + yj * (Ei - Ej) / eta, low, high)
                    if abs(new_aj - aj) < 1e-6 * (new_aj + aj + 1e-6):
                        continue
                    new_ai = ai + yi * yj * (aj - new_aj)

                    delta_ai = new_ai - ai
                    delta_aj = new_aj - aj
                    b1 = bias - Ei - yi * delta_ai * gram[i, i] - yj * delta_aj * gram[i, j]
                    b2 = bias - Ej - yi * delta_ai * gram[i, j] - yj * delta_aj * gram[j, j]
                    if 1e-12 < new_ai < self.C - 1e-12:
                        new_bias = b1
                    elif 1e-12 < new_aj < self.C - 1e-12:
                        new_bias = b2
                    else:
                        new_bias = 0.5 * (b1 + b2)

                    errors += (
                        yi * delta_ai * gram[i]
                        + yj * delta_aj * gram[j]
                        + (new_bias - bias)
                    )
                    alphas[i], alphas[j], bias = new_ai, new_aj, new_bias
                    changed += 1
                    updates += 1
                    updated = True
                    break
                if not updated:
                    continue

            passes_without_change = 0 if changed else passes_without_change + 1

        support = alphas > 1e-8
        self.support_vectors_ = X[support].copy()
        self.support_labels_ = y_signed[support].copy()
        self.dual_coef_ = (alphas[support] * y_signed[support]).copy()
        self.intercept_ = float(bias)
        self.n_iter_ = int(updates)
        self.n_examined_ = int(examined)
        self.n_support_ = int(support.sum())
        self.converged_ = passes_without_change >= self.max_passes
        self.is_fitted_ = True
        return self

    def decision_function(self, X, *, batch_size=256):
        if not self.is_fitted_:
            raise ValueError("Fit the model before prediction")
        X = self._as_2d_finite(X, name="X")
        if not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same feature count used in training")
        outputs = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            K = self._kernel_matrix(X[start:stop], self.support_vectors_)
            outputs[start:stop] = K @ self.dual_coef_ + self.intercept_
        return outputs

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def save(self, path):
        if not self.is_fitted_:
            raise ValueError("Fit the model before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, KernelSVM) or not model.is_fitted_:
            raise ValueError("File does not contain a fitted KernelSVM")
        return model
