# Estimasi mutual information kontinu: Ross (2014), https://doi.org/10.1371/journal.pone.0087357.
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma


class KNNSVMPreprocessor:
    # Statistik dan bobot fitur dipelajari hanya dari data training.

    def __init__(self, exponent=0.5, min_score=0.002, n_bins=10,
                 estimator="quantile", n_neighbors=5, random_state=42,
                 categorical_mi="dummy"):
        if (exponent < 0 or min_score <= 0 or n_bins < 2 or n_neighbors < 1
                or estimator not in ("quantile", "knn")
                or categorical_mi not in ("dummy", "group")):
            raise ValueError("Invalid MI preprocessing configuration")
        self.exponent = exponent
        self.min_score = min_score
        self.n_bins = int(n_bins)
        self.estimator = estimator
        self.n_neighbors = int(n_neighbors)
        self.random_state = random_state
        self.categorical_mi = categorical_mi

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates or X.shape[1] == 0:
            raise ValueError("X must be a DataFrame with unique feature columns")
        y = np.asarray(y)
        if len(X) == 0 or y.ndim != 1 or len(y) != len(X) or pd.isna(y).any():
            raise ValueError("X and y must be nonempty and have matching lengths")
        _, y_codes = np.unique(y, return_inverse=True)
        if np.unique(y_codes).size < 2:
            raise ValueError("y must contain at least two classes")
        self.columns_ = X.columns.tolist()
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_ = [column for column in self.columns_ if column not in self.numeric_]
        self.medians_, self.means_, self.scales_ = {}, {}, {}
        self.modes_, self.categories_ = {}, {}
        for column in self.numeric_:
            values = pd.to_numeric(X[column], errors="raise").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, np.nan)
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            filled = np.where(np.isnan(values), median, values)
            self.medians_[column] = median
            self.means_[column] = float(np.mean(filled))
            scale = float(np.std(filled))
            self.scales_[column] = scale if scale > 0 else 1.0
        for column in self.categorical_:
            values = X[column].astype("string")
            counts = values.value_counts(dropna=True)
            # Urutkan modus yang sama frekuensinya agar hasil konsisten.
            mode = sorted(counts[counts == counts.max()].index.tolist())[0] if len(counts) else ""
            self.modes_[column] = mode
            self.categories_[column] = sorted(values.fillna(mode).unique().tolist())

        matrix = self._base_transform(X)
        unique_counts = np.array([np.unique(matrix[:, i]).size for i in range(matrix.shape[1])])
        continuous_indices = np.flatnonzero(unique_counts >= 20)
        # Tambahkan derau kecil agar jarak yang sama dapat dibedakan.
        continuous_values = matrix[:, continuous_indices].copy()
        if continuous_values.shape[1] and self.estimator == "knn":
            scales = np.std(continuous_values, axis=0)
            scales[scales == 0] = 1.0
            continuous_values /= scales
            noise_scale = np.maximum(1., np.mean(np.abs(continuous_values), axis=0))
            rng = np.random.RandomState(self.random_state)
            continuous_values += 1e-10 * noise_scale * rng.standard_normal(continuous_values.shape)
        continuous_position = {col: pos for pos, col in enumerate(continuous_indices)}
        scores, edges = [], []
        for i in range(matrix.shape[1]):
            values = matrix[:, i]
            if i in continuous_position and self.estimator == "knn":
                scores.append(self._continuous_mi(continuous_values[:, continuous_position[i]], y_codes))
                edges.append(None)
                continue
            if unique_counts[i] < 20:
                bins = None
                _, codes = np.unique(values, return_inverse=True)
            else:
                bins = np.unique(np.quantile(values, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                codes = np.searchsorted(bins, values, side="right")
            n_codes, n_classes = int(np.max(codes)) + 1, int(np.max(y_codes)) + 1
            counts = np.bincount(codes * n_classes + y_codes,
                                 minlength=n_codes * n_classes).reshape(n_codes, n_classes)
            joint = counts / len(y)
            independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
            nz = joint > 0
            scores.append(float(np.sum(joint[nz] * np.log(joint[nz] / independent[nz]))))
            edges.append(bins)
        self.bin_edges_ = edges
        scores_arr = np.maximum(np.array(scores, dtype=float), 0.0)
        if getattr(self, "categorical_mi", "dummy") == "group":
            # Gunakan MI tingkat kolom agar fitur kategori multi-nilai tidak terdilusi one-hot.
            col_ptr = len(self.numeric_)
            n_classes = int(np.max(y_codes)) + 1
            for column in self.categorical_:
                n_cat = len(self.categories_[column])
                vals = X[column].astype("string").fillna(self.modes_[column]).to_numpy(dtype=str)
                _, codes = np.unique(vals, return_inverse=True)
                n_codes = int(np.max(codes)) + 1
                counts = np.bincount(codes * n_classes + y_codes,
                                     minlength=n_codes * n_classes).reshape(n_codes, n_classes)
                joint = counts / len(y)
                independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
                nz = joint > 0
                group_mi = float(max(0.0, np.sum(joint[nz] * np.log(joint[nz] / independent[nz]))))
                scores_arr[col_ptr:col_ptr + n_cat] = group_mi
                col_ptr += n_cat
        self.scores_ = scores_arr
        mean_score = float(np.mean(self.scores_))
        self.weights_ = ((np.maximum(self.scores_, self.min_score) / mean_score) ** self.exponent
                         if mean_score > 0 else np.ones(matrix.shape[1]))
        self.n_features_in_ = len(self.columns_)
        self.n_features_out_ = matrix.shape[1]
        return self

    def _continuous_mi(self, values, labels):
        # Hitung tetangga dalam radius tetangga kelas yang sama.
        
        eligible = np.zeros(len(values), dtype=bool)
        radius = np.zeros(len(values))
        label_count = np.zeros(len(values), dtype=int)
        neighbors = np.zeros(len(values), dtype=int)
        for label in np.unique(labels):
            selected = np.flatnonzero(labels == label)
            count = len(selected)
            if count <= 1:
                continue
            k = min(self.n_neighbors, count - 1)
            points = values[selected, None]
            distances, _ = cKDTree(points).query(points, k=k + 1)
            radius[selected] = np.nextafter(distances[:, -1], 0.0)
            label_count[selected] = count
            neighbors[selected] = k
            eligible[selected] = True
        if not eligible.any():
            return 0.0
        points = values[eligible, None]
        counts = cKDTree(points).query_ball_point(points, radius[eligible], return_length=True)
        estimate = (digamma(len(points)) + np.mean(digamma(neighbors[eligible]))
                    - np.mean(digamma(label_count[eligible])) - np.mean(digamma(counts)))
        return float(max(0.0, estimate))

    def _base_transform(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates or set(X.columns) != set(self.columns_):
            raise ValueError("Feature columns must match fitted columns")
        result = []
        for column in self.numeric_:
            values = pd.to_numeric(X[column], errors="raise").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.medians_[column])
            result.append(((values - self.means_[column]) / self.scales_[column])[:, None])
        for column in self.categorical_:
            values = X[column].astype("string").fillna(self.modes_[column]).to_numpy(dtype=str)
            result.append((values[:, None] == np.asarray(self.categories_[column], dtype=str)[None, :]).astype(float))
        return np.concatenate(result, axis=1) if result else np.empty((len(X), 0))

    def transform(self, X):
        if not hasattr(self, "weights_"):
            raise ValueError("Preprocessor is not fitted")
        return self._base_transform(X) * self.weights_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    # Uji pembobotan MI kategori tingkat grup agar semua indikator satu kolom berbobot sama.
    df_uji = pd.DataFrame({
        "angka": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "grup": ["A", "A", "B", "B", "C", "C"],
    })
    y_uji = np.array([0, 0, 1, 1, 1, 0])
    p_grp = KNNSVMPreprocessor(categorical_mi="group").fit(df_uji, y_uji)
    assert np.allclose(p_grp.scores_[1:], p_grp.scores_[1])
    assert p_grp.transform(df_uji).shape == (6, 4)
    print("Uji KNNSVMPreprocessor (categorical_mi='group') lolos")
