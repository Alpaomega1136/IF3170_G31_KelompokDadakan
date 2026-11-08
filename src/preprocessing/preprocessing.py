from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
