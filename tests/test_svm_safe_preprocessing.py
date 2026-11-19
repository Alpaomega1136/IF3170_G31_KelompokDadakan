import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing.svm_safe_preprocessing import SafeSVMPreprocessor


class SafePreprocessingTest(unittest.TestCase):
    def test_fit_transform_uses_only_training_statistics(self):
        train = pd.DataFrame({"number": [1., 2., np.nan, 5.],
                              "category": ["b", "a", None, "b"]})
        valid = pd.DataFrame({"number": [100., np.nan], "category": ["new", None]})
        prep = SafeSVMPreprocessor(exponent=0)
        transformed = prep.fit_transform(train, np.array([0, 1, 0, 1]))
        self.assertEqual(prep.medians_["number"], 2.)
        self.assertEqual(prep.modes_["category"], "b")
        self.assertEqual(prep.categories_["category"], ["a", "b"])
        np.testing.assert_allclose(transformed[:, 0].mean(), 0, atol=1e-12)
        result = prep.transform(valid)
        np.testing.assert_array_equal(result[0, 1:], [0, 0])
        np.testing.assert_array_equal(result[1, 1:], [0, 1])
        self.assertEqual(prep.medians_["number"], 2.)
        np.testing.assert_allclose(result, pickle.loads(pickle.dumps(prep)).transform(valid))

    def test_mutual_information_and_input_validation(self):
        X = pd.DataFrame({"informative": [0, 0, 1, 1] * 10,
                          "constant": [5] * 40})
        y = np.array([0, 0, 1, 1] * 10)
        prep = SafeSVMPreprocessor().fit(X, y)
        self.assertAlmostEqual(prep.scores_[0], np.log(2))
        self.assertAlmostEqual(prep.scores_[1], 0)
        with self.assertRaises(ValueError):
            prep.transform(X.drop(columns="constant"))
        with self.assertRaises(ValueError):
            SafeSVMPreprocessor().fit(X, np.zeros(len(X)))

    def test_knn_mi_matches_reference_for_ties_and_mixed_columns(self):
        from sklearn.feature_selection import mutual_info_classif

        rng = np.random.RandomState(123)
        y = rng.randint(0, 2, size=200)
        X = pd.DataFrame({"continuous": np.round(rng.normal(y, 0.8), 1),
                          "binary": rng.randint(0, 2, size=200),
                          "category": np.where(y, "yes", "no")})
        prep = SafeSVMPreprocessor(estimator="knn", random_state=42).fit(X, y)
        base = prep._base_transform(X)
        discrete = np.array([np.unique(base[:, i]).size < 20 for i in range(base.shape[1])])
        for column in np.flatnonzero(discrete):
            _, base[:, column] = np.unique(base[:, column], return_inverse=True)
        reference = mutual_info_classif(base, y, discrete_features=discrete,
                                       n_neighbors=5, random_state=42)
        np.testing.assert_allclose(prep.scores_, reference, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
