import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def impute(X_train, X_valid, X_test):
    """Fill missing values using statistics fitted on training data."""
    if not X_train.columns.equals(X_valid.columns) or not X_train.columns.equals(X_test.columns):
        raise ValueError("Kolom training, validasi, dan test harus sama")
    frames = [frame.copy() for frame in (X_train, X_valid, X_test)]
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    imputers = {}
    for name, columns, strategy in [("numerik", numeric, "median"),
                                    ("kategori", categorical, "most_frequent")]:
        if not columns:
            continue
        if X_train[columns].isna().all().any():
            raise ValueError("Kolom training tidak boleh seluruhnya kosong")
        imputer = SimpleImputer(strategy=strategy).fit(X_train[columns])
        for frame in frames:
            frame[columns] = imputer.transform(frame[columns])
        imputers[name] = imputer
    return frames[0], frames[1], frames[2], imputers


def encode(X_train, X_valid, X_test):
    """One-hot encode categories using only training categories."""
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    if not categorical:
        return X_train.copy(), X_valid.copy(), X_test.copy(), None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[categorical])
    frames = []
    for frame in (X_train, X_valid, X_test):
        encoded = pd.DataFrame(encoder.transform(frame[categorical]), index=frame.index,
                               columns=encoder.get_feature_names_out(categorical))
        if encoded.columns.isin(frame.drop(columns=categorical).columns).any():
            raise ValueError("Nama kolom hasil encoding bertabrakan dengan kolom lain")
        frames.append(pd.concat([frame.drop(columns=categorical), encoded], axis=1))
    return frames[0], frames[1], frames[2], encoder


def scale(X_train, X_valid, X_test, columns):
    """Standardize numeric columns using training statistics."""
    frames = [frame.copy() for frame in (X_train, X_valid, X_test)]
    if len(columns) == 0:
        return frames[0], frames[1], frames[2], None
    scaler = StandardScaler().fit(X_train[columns])
    for frame in frames:
        frame[columns] = scaler.transform(frame[columns])
    return frames[0], frames[1], frames[2], scaler


def create_preprocessor(data_train):
    kolom_numerik = data_train.select_dtypes(include="number").columns.tolist()
    kolom_kategori = data_train.select_dtypes(exclude="number").columns.tolist()

    numerik = Pipeline([
        ("imputasi", SimpleImputer(strategy="median")),
        ("standardisasi", StandardScaler()),
    ])
    kategori = Pipeline([
        ("imputasi", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numerik", numerik, kolom_numerik),
        ("kategori", kategori, kolom_kategori),
    ])


class LRFeaturePreprocessor:
    """Add domain nonlinearities for Logistic Regression, then impute/scale/encode on train only."""

    @staticmethod
    def _augment(X):
        out = X.copy()
        out["year_month"] = (
            out["arrival_year"].astype(str) + "-" + out["arrival_month"].astype(str).str.zfill(2)
        )
        out["special_requests_isna"] = out["special_requests"].isna().astype(float)
        out["special_requests_cat"] = (
            out["special_requests"].fillna(-1).clip(upper=3).astype(int).astype(str)
        )
        out["rate_is_zero"] = (out["nightly_rate_idr"] == 0).astype(float)
        out["total_nights_zero"] = ((out["weekend_nights"] + out["weekday_nights"]) == 0).astype(float)
        out["log_dba"] = np.log1p(out["days_before_arrival"])
        out["log_prior_comp"] = np.log1p(out["prior_completed_bookings"])
        out["log_prior_canc"] = np.log1p(out["prior_cancellations"])
        out["has_prior_comp"] = (out["prior_completed_bookings"] > 0).astype(float)
        out["ota_no_req"] = (
            (out["booking_channel"] == "Online Travel Agent")
            & (out["special_requests"].fillna(0) == 0)
        ).astype(float)
        out["dba_gt_150"] = (out["days_before_arrival"] > 150).astype(float)
        sr_bucket = np.where(
            out["special_requests"].isna(), "NA",
            np.where(out["special_requests"] == 0, "0", "pos")
        )
        out["ch_x_sr"] = out["booking_channel"].astype(str) + "_" + sr_bucket
        return out

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        augmented = self._augment(X[self.columns_])
        self.transformer_ = create_preprocessor(augmented)
        self.transformer_.fit(augmented, y)
        return self

    def transform(self, X):
        if not hasattr(self, "transformer_"):
            raise ValueError("Fit the preprocessor before transform")
        return self.transformer_.transform(self._augment(X[self.columns_]))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
