import pickle
import numpy as np


class KNNFromScratch:
    def __init__(self, k=5, metric='euclidean', weights='uniform', threshold=None, batch_size=128):
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError('metric must be euclidean or manhattan')
        if weights not in ('uniform', 'distance'):
            raise ValueError('weights must be uniform or distance')
        if threshold is not None and not (0.0 < threshold < 1.0):
            raise ValueError('threshold must be between 0 and 1')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.k = k
        self.metric = metric
        self.weights = weights
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
        self.X_train = np.ascontiguousarray(X)
        self._X_train_T = np.ascontiguousarray(X.T)
        self.classes_, self._y_encoded = np.unique(y, return_inverse=True)
        self.y_train = y.copy()
        self.n_features_in_ = X.shape[1]
        self._train_sq_norm = np.sum(self.X_train ** 2, axis=1)
        return self

    def _batch_distances(self, X_batch, dist_buf=None, work_buf=None):
        if self.metric == 'euclidean':
            batch_sq = np.sum(X_batch ** 2, axis=1)[:, None]
            sq = np.maximum(batch_sq + self._train_sq_norm[None, :] - 2.0 * (X_batch @ self.X_train.T), 0.0)
            return np.sqrt(sq)
        # Gunakan buffer agar pemakaian memori tetap terbatas per batch.
        b = len(X_batch)
        Xt = getattr(self, '_X_train_T', None)
        if Xt is None:
            Xt = np.ascontiguousarray(self.X_train.T)
        dist = dist_buf[:b] if dist_buf is not None else np.empty((b, len(self.X_train)), dtype=np.float64)
        buf = work_buf[:b] if work_buf is not None else np.empty((b, len(self.X_train)), dtype=np.float64)
        dist.fill(0.0)
        for j in range(self.n_features_in_):
            np.subtract(X_batch[:, j:j + 1], Xt[j], out=buf)
            np.abs(buf, out=buf)
            dist += buf
        return dist

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
        probs = np.empty((len(X), n_classes), dtype=np.float64)
        dist_buf = np.empty((min(batch_size, len(X)), n_train), dtype=np.float64) if self.metric == 'manhattan' else None
        work_buf = np.empty((min(batch_size, len(X)), n_train), dtype=np.float64) if self.metric == 'manhattan' else None

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            dist = self._batch_distances(X[start:end], dist_buf, work_buf)
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
                zero_mask = dk == 0.0
                has_zero = zero_mask.any(axis=1, keepdims=True)
                w = np.where(has_zero, zero_mask.astype(np.float64), 1.0 / np.where(zero_mask, 1.0, dk))
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
    print("Uji KNNFromScratch (deterministic tie-break) lolos")
