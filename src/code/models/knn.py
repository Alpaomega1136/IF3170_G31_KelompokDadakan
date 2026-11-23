import pickle
import numpy as np
from scipy.spatial.distance import cdist


class KNNFromScratch:
    def __init__(self, k=5, metric='euclidean', weights='uniform',
                 weight_power=1.0, weight_eps=0.0, local_scale_k=0,
                 local_scale_alpha=0.0, local_scale_eps=0.1,
                 threshold=None, batch_size=128):
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError('metric must be euclidean or manhattan')
        if weights not in ('uniform', 'distance'):
            raise ValueError('weights must be uniform or distance')
        if not isinstance(weight_power, (int, float)) or weight_power <= 0:
            raise ValueError('weight_power must be positive')
        if not isinstance(weight_eps, (int, float)) or weight_eps < 0:
            raise ValueError('weight_eps must be non-negative')
        if not isinstance(local_scale_k, int) or local_scale_k < 0:
            raise ValueError('local_scale_k must be a non-negative integer')
        if not isinstance(local_scale_alpha, (int, float)) or local_scale_alpha < 0:
            raise ValueError('local_scale_alpha must be non-negative')
        if not isinstance(local_scale_eps, (int, float)) or local_scale_eps <= 0:
            raise ValueError('local_scale_eps must be positive')
        if local_scale_alpha > 0 and local_scale_k < 1:
            raise ValueError('local_scale_k must be positive when local_scale_alpha > 0')
        if threshold is not None and not (0.0 < threshold < 1.0):
            raise ValueError('threshold must be between 0 and 1')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.k = k
        self.metric = metric
        self.weights = weights
        self.weight_power = float(weight_power)
        self.weight_eps = float(weight_eps)
        self.local_scale_k = local_scale_k
        self.local_scale_alpha = float(local_scale_alpha)
        self.local_scale_eps = float(local_scale_eps)
        self.threshold = threshold
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0 or X.shape[1] == 0:
            raise ValueError('X_train must be a nonempty 2D array matching y_train')
        if not np.isfinite(X).all():
            raise ValueError('X_train must contain only finite numbers')
        if self.k > len(X):
            raise ValueError('k cannot exceed the number of training samples')
        if self.local_scale_alpha > 0.0 and self.local_scale_k >= len(X):
            raise ValueError('local_scale_k must be smaller than the number of training samples')
        self.X_train = np.ascontiguousarray(X)
        self.classes_, self._y_encoded = np.unique(y, return_inverse=True)
        self.y_train = y.copy()
        self.n_features_in_ = X.shape[1]
        self._inv_local_scale_ = self._fit_local_scale() if self.local_scale_alpha > 0.0 else None
        return self

    def _batch_distances(self, X_batch):
        cdist_metric = 'cityblock' if self.metric == 'manhattan' else 'euclidean'
        return cdist(X_batch, self.X_train, metric=cdist_metric)

    def _fit_local_scale(self):
        # Local scaling berbasis radius tetangga training untuk mengurangi hubness
        # (Zelnik-Manor & Perona, 2004; Schnitzer dkk., 2012).
        n_train = len(self.X_train)
        m = self.local_scale_k
        radii = np.empty(n_train, dtype=np.float64)
        for start in range(0, n_train, self.batch_size):
            end = min(start + self.batch_size, n_train)
            dist = self._batch_distances(self.X_train[start:end])
            # Abaikan jarak ke titik sendiri agar radius lokal dihitung dari tetangga lain.
            dist[np.arange(end - start), np.arange(start, end)] = np.inf
            radii[start:end] = np.partition(dist, m - 1, axis=1)[:, :m].mean(axis=1)
        scale = (radii + self.local_scale_eps) ** self.local_scale_alpha
        return float(np.mean(scale)) / scale

    def predict_proba(self, X_test):
        if not hasattr(self, 'X_train'):
            raise ValueError('Fit the model before prediction')
        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('X_test must be finite with the same number of features used in training')
        n_classes = len(self.classes_)
        if len(X) == 0:
            return np.empty((0, n_classes), dtype=np.float64)

        k = self.k
        n_train = len(self.X_train)
        batch_size = getattr(self, 'batch_size', 128)
        weights_mode = getattr(self, 'weights', 'uniform')
        inv_scale = getattr(self, '_inv_local_scale_', None)
        probs = np.empty((len(X), n_classes), dtype=np.float64)

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            dist = self._batch_distances(X[start:end])
            if inv_scale is not None:
                dist *= inv_scale[None, :]
            rows = np.arange(end - start)[:, None]
            if k < n_train:
                idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
                # Jika jarak sama, dahulukan indeks data training yang lebih kecil.
                d_k = np.max(dist[rows, idx], axis=1, keepdims=True)
                tied_rows = np.flatnonzero((dist <= d_k).sum(axis=1) > k)
                for r in tied_rows:
                    threshold_d = d_k[r, 0]
                    closer = np.flatnonzero(dist[r] < threshold_d)
                    boundary = np.flatnonzero(dist[r] == threshold_d)
                    idx[r] = np.concatenate([closer, boundary[:k - len(closer)]])
            else:
                idx = np.broadcast_to(np.arange(n_train), (end - start, n_train))
            dk = dist[rows, idx]
            yk = self._y_encoded[idx]

            if weights_mode == 'distance':
                # Pembobotan jarak Shepard terregularisasi (Shepard, 1968): 1 / (d + eps)^p.
                power = getattr(self, 'weight_power', 1.0)
                eps = getattr(self, 'weight_eps', 0.0)
                if eps > 0.0:
                    w = 1.0 / ((dk + eps) ** power)
                else:
                    zero_mask = dk == 0.0
                    has_zero = zero_mask.any(axis=1, keepdims=True)
                    denom = np.where(zero_mask, 1.0, dk) ** power
                    w = np.where(has_zero, zero_mask.astype(np.float64), 1.0 / denom)
            else:
                w = np.ones_like(dk, dtype=np.float64)

            totals = np.zeros((end - start, n_classes), dtype=np.float64)
            for c in range(n_classes):
                totals[:, c] = np.where(yk == c, w, 0.0).sum(axis=1)
            probs[start:end] = totals / totals.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X_test, threshold=None):
        probs = self.predict_proba(X_test)
        th = getattr(self, 'threshold', None) if threshold is None else threshold
        if th is not None:
            if len(self.classes_) != 2:
                raise ValueError('threshold only works for binary classification')
            return np.where(probs[:, 1] >= th, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(probs, axis=1)]

    def save(self, path):
        if not hasattr(self, 'X_train'):
            raise ValueError('Fit the model before saving')
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


if __name__ == "__main__":
    # Uji pemilihan tetangga dengan jarak yang sama.
    X_tie = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    y_tie = np.array([1, 1, 0, 0, 0])
    q = np.array([[0.0, 0.0]])
    for metric in ("euclidean", "manhattan"):
        m = KNNFromScratch(k=3, metric=metric, weights="uniform").fit(X_tie, y_tie)
        # Tetangga berjarak sama dipilih berdasarkan urutan indeks.
        # Mayoritas label tetangga terpilih adalah kelas satu.
        assert m.predict(q)[0] == 1
    # Uji bobot Shepard terregularisasi agar satu titik dekat tidak mendominasi tetangga lain.
    X_reg = np.array([[0.0], [0.05], [0.05], [0.05], [0.05]])
    y_reg = np.array([1, 0, 0, 0, 0])
    p_reg = KNNFromScratch(
        k=5, metric="manhattan", weights="distance", weight_power=2.5, weight_eps=0.3
    ).fit(X_reg, y_reg).predict_proba([[0.0]])
    assert np.isfinite(p_reg).all() and p_reg[0, 0] > 0.7
    # Uji local scaling: klaster padat (radius lokal kecil) tidak mendominasi titik jarang.
    X_ls = np.array([[-0.01], [0.0], [0.0], [0.01], [1.1], [2.5], [4.0]])
    y_ls = np.array([0, 0, 0, 0, 1, 1, 1])
    q_ls = np.array([[0.52]])
    assert KNNFromScratch(k=1, metric="manhattan").fit(X_ls, y_ls).predict(q_ls)[0] == 0
    assert KNNFromScratch(
        k=1, metric="manhattan", local_scale_k=2, local_scale_alpha=0.5, local_scale_eps=0.05
    ).fit(X_ls, y_ls).predict(q_ls)[0] == 1
    print("Uji KNNFromScratch (tie-break, Shepard terregularisasi, local scaling) lolos")
