import pickle
from numbers import Real

import numpy as np


class NaiveBayes:
    def __init__(self, categorical_features=None, m=None, var_smoothing=1e-9, threshold=None):
        self.categorical_features = categorical_features
        self.m = m
        self.var_smoothing = var_smoothing
        self.threshold = threshold
    @staticmethod
    def _missing(value):
        return value is None or (isinstance(value, Real) and np.isnan(value))

    def _missing_mask(self, column):
        return np.fromiter((self._missing(v) for v in column), bool, len(column))

    
    def fit(self, X, y):
        self.feature_names_= list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X,dtype=object)
        y = np.asarray(y)
        if X.ndim!=2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D array and y must match its rows")
        if any(self._missing(v) for v in y):
            raise ValueError("y must not contain missing labels")
        if self.feature_names_ is None:
            self.feature_names_ = list(range(X.shape[1]))

        self.classes_, encoded= np.unique(y, return_inverse=True)
        counts = np.bincount(encoded)
        self.class_log_prior_ =np.log(counts / counts.sum())

        if self.categorical_features is None:
            self.is_categorical_= [True] * X.shape[1]
        else:
            unknown = set(self.categorical_features) - set(self.feature_names_)
            if unknown:
                raise ValueError(f"Unknown categorical features: {unknown}")
            self.is_categorical_ = [name in self.categorical_features for name in self.feature_names_]

        self.params_ = []
        for j in range(X.shape[1]):
            if self.is_categorical_[j]:
                self.params_.append(self._fit_categorical(X[:, j], encoded))
            else:
                self.params_.append(self._fit_gaussian(X[:, j], encoded))

        gaussian = [p for p, cat in zip(self.params_, self.is_categorical_) if not cat]
        if gaussian:
            max_var = max(p["total_var"] for p in gaussian)
            # Jaga variansi tetap positif saat fitur konstan atau seluruhnya kosong.
            epsilon = self.var_smoothing * max_var if max_var > 0 else max(self.var_smoothing, 1e-12)
            for p in gaussian:
                p["var"] = np.maximum(p["var"] + epsilon, 1e-12)
        return self

    def _fit_categorical(self, column, y_encoded):
        missing = self._missing_mask(column)
        values, codes = np.unique(column[~missing], return_inverse=True)
        y_known = y_encoded[~missing]

        n_classes, K = len(self.classes_),len(values)
        counts = np.zeros((n_classes, K))
        np.add.at(counts, (y_known, codes), 1)

        m = K if self.m is None else self.m
        p = 1 / K if K > 0 else 0.0
        n_c = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prob = np.log((counts + m * p) / (n_c + m))

        return {"values": values,
                "index": {v: i for i, v in enumerate(values)},
                "log_prob": log_prob}


    def _fit_gaussian(self, column, y_encoded):
        missing = self._missing_mask(column)
        x = column[~missing].astype(float)
        y_known = y_encoded[~missing]
        n_classes = len(self.classes_)
        overall_mean = float(x.mean()) if len(x) else 0.0
        overall_var = float(x.var()) if len(x) else 0.0
        mean = np.zeros(n_classes)
        var = np.zeros(n_classes)
        for c in range(n_classes):
            x_c = x[y_known == c]
            mean[c] = x_c.mean() if len(x_c) else overall_mean
            var[c] = x_c.var() if len(x_c) else overall_var
        return {"mean": mean, "var": var, "total_var": overall_var}


    def _joint_log_likelihood(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("Fit the model before prediction")
        if hasattr(X, "columns") and list(X.columns) != self.feature_names_:
            raise ValueError("X must have the same columns used in training")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1]!= len(self.params_):
            raise ValueError("X must have the same number of features used in training")

        jll = np.tile(self.class_log_prior_, (len(X), 1))
        for j, params in enumerate(self.params_):
            column=X[:, j]
            missing = self._missing_mask(column)
            if self.is_categorical_[j]:
                idx =np.array([-1 if miss else params["index"].get(v, -1) for v, miss in zip(column, missing)])
                known = idx >= 0
                jll[known] += params["log_prob"][:, idx[known]].T
            else:
                known = ~missing
                x = column[known].astype(float)[:, None]
                mean, var = params["mean"], params["var"]
                jll[known] += -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)
        return jll


    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        top = jll.max(axis=1, keepdims=True)
        log_norm = top + np.log(np.exp(jll - top).sum(axis=1, keepdims=True))
        return jll - log_norm


    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))


    def predict(self, X, threshold=None):
        th = self.threshold if threshold is None else threshold
        if th is None:
            return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
        if len(self.classes_) != 2:
            raise ValueError("threshold only works for binary classification")
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= th, self.classes_[1], self.classes_[0])


    def save(self, path):
        if not hasattr(self,"classes_"):
            raise ValueError("Fit the model before saving")
        with open(path,"wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as  file:
            return  pickle.load(file)


if __name__ == "__main__":
    import pandas as pd
    play_tennis = pd.DataFrame(
        [
            ["Sunny", "Hot", "High", "Weak", "No"],
            ["Sunny", "Hot", "High", "Strong", "No"],
            ["Overcast", "Hot", "High", "Weak", "Yes"],
            ["Rain", "Mild", "High", "Weak", "Yes"],
            ["Rain", "Cool", "Normal", "Weak", "Yes"],
            ["Rain", "Cool", "Normal", "Strong", "No"],
            ["Overcast", "Cool", "Normal", "Strong", "Yes"],
            ["Sunny", "Mild", "High", "Weak", "No"],
            ["Sunny", "Cool", "Normal", "Weak", "Yes"],
            ["Rain", "Mild", "Normal", "Weak", "Yes"],
            ["Sunny", "Mild", "Normal", "Strong", "Yes"],
            ["Overcast", "Mild", "High", "Strong", "Yes"],
            ["Overcast", "Hot", "Normal", "Weak", "Yes"],
            ["Rain", "Mild", "High", "Strong", "No"],
        ],
        columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"],
    )
    X = play_tennis.drop(columns="PlayTennis")
    y = play_tennis["PlayTennis"]
    query = pd.DataFrame([["Sunny", "Cool", "High", "Strong"]], columns=X.columns)

    model = NaiveBayes(m=0).fit(X, y)
    jll = np.exp(model._joint_log_likelihood(query))[0]
    proba = model.predict_proba(query)[0]
    print("classes          :", model.classes_)
    print("P(c)*prod P(x|c) :", jll.round(4))
    print("P(c | x)         :", proba.round(3))
    print("prediksi         :", model.predict(query)[0])
    assert np.allclose(jll, [0.0206, 0.0053], atol=1e-4)
    assert model.predict(query)[0] == "No"

    # Uji fitur Gaussian yang konstan dan seluruhnya kosong.
    for col in [[[1.0], [1.0], [1.0], [1.0]], [[np.nan], [np.nan], [np.nan], [np.nan]]]:
        arr = np.array(col, dtype=object)
        probs = NaiveBayes(categorical_features=[]).fit(arr, np.array([0, 0, 1, 1])).predict_proba(arr)
        assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0)
    print("Uji PlayTennis dan edge-case Gaussian lolos")
