"""Checks for SVM dual optimization, including its serialization contract."""
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.svm_working_set import WorkingSetSVM
from preprocessing.svm_feature_weights import MutualInformationWeights


class WorkingSetSVMTests(unittest.TestCase):
    def test_kernels_match_reference_and_satisfy_kkt(self):
        X, y = make_moons(n_samples=160, noise=0.2, random_state=10)
        for kernel in ("linear", "polynomial", "rbf"):
            with self.subTest(kernel=kernel):
                model = WorkingSetSVM(C=1.2, kernel=kernel, gamma=0.5, coef0=1,
                                      degree=3, tol=1e-5).fit(X, y)
                reference = SVC(C=1.2, kernel="poly" if kernel == "polynomial" else kernel,
                                gamma=0.5, coef0=1, degree=3, tol=1e-5).fit(X, y)
                self.assertTrue(model.converged_)
                self.assertLess(model.kkt_gap_, 1e-5)
                np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X),
                                           atol=2e-4)
                np.testing.assert_array_equal(model.predict(X), reference.predict(X))

    def test_duplicate_points_with_conflicting_labels_and_pickle(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [3, 3]], dtype=float)
        y = np.array([0, 1, 0, 0, 1, 1])
        model = WorkingSetSVM(C=2, gamma=0.3, tol=1e-5, cache_mb=0.001).fit(X, y)
        reference = SVC(C=2, gamma=0.3, tol=1e-5).fit(X, y)
        self.assertTrue(model.converged_)
        np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X),
                                   atol=2e-4)
        loaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_invalid_inputs(self):
        model = WorkingSetSVM()
        with self.assertRaises(ValueError):
            model.fit([[0], [1]], [1, 1])
        with self.assertRaises(ValueError):
            model.fit([[0], [np.nan]], [0, 1])

    def test_feature_weights_are_fitted_and_reused_for_validation(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(100, 3))
        y = (X[:, 0] > 0).astype(int)
        weights = MutualInformationWeights().fit(X, y)
        saved_weights = weights.weights_.copy()
        validation = rng.normal(size=(20, 3))
        np.testing.assert_allclose(weights.transform(validation), validation * saved_weights)
        np.testing.assert_array_equal(weights.weights_, saved_weights)
        self.assertGreater(weights.weights_[0], weights.weights_[1])


if __name__ == "__main__":
    unittest.main()
