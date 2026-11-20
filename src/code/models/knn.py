import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError('metric must be euclidean or manhattan')
        if weights not in ('uniform', 'distance'):
            raise ValueError('weights must be uniform or distance')
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        tebakann = [self._prediksi_1_baris(x) for x in X_test]
        return np.array(tebakann)

    def _prediksi_1_baris(self, x): 
        if self.metric == 'manhattan':
            jarakk = np.sum(np.abs(self.X_train - x), axis=1) #manhattan
        else:
            jarakk = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1)) #euclidean 

        indeks_k_plg_dekat = np.argsort(jarakk)[:self.k]
        label_k_plg_dekat = [self.y_train[i] for i in indeks_k_plg_dekat]
        if getattr(self, 'weights', 'uniform') == 'distance':
            jarak_terdekat = jarakk[indeks_k_plg_dekat]
            nol = jarak_terdekat == 0
            bobot = nol.astype(float) if nol.any() else 1 / jarak_terdekat
            nilai = np.unique(label_k_plg_dekat)
            total = [bobot[np.asarray(label_k_plg_dekat) == label].sum() for label in nilai]
            return nilai[np.argmax(total)]
        suara_plg_banyak = Counter(label_k_plg_dekat).most_common(1)
        return suara_plg_banyak[0][0]
