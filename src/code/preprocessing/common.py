import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def impute(X_train, X_valid, X_test):
    # Isi nilai kosong berdasarkan statistik data training.
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
    # Pelajari kategori hanya dari data training.
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
    # Pelajari parameter standardisasi hanya dari data training.
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
