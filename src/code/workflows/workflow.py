import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_valid, y_valid, threshold=None):
    prediction = (model.predict(X_valid) if threshold is None else
                  (model.decision_function(X_valid) >= threshold).astype(int))
    scores = pd.Series({
        "accuracy": accuracy_score(y_valid, prediction),
        "precision_kelas_1": precision_score(y_valid, prediction, zero_division=0),
        "recall_kelas_1": recall_score(y_valid, prediction, zero_division=0),
        "f1_kelas_0": f1_score(y_valid, prediction, pos_label=0, zero_division=0),
        "f1_kelas_1": f1_score(y_valid, prediction, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_valid, prediction, average="macro", labels=[0, 1], zero_division=0),
    })
    matrix = pd.DataFrame(confusion_matrix(y_valid, prediction, labels=[0, 1]),
                          index=["aktual 0", "aktual 1"], columns=["prediksi 0", "prediksi 1"])
    return scores, matrix


def compare_models(model_scratch, model_sklearn, X_train, y_train, X_valid, y_valid, threshold=None):
    train_model(model_sklearn, X_train, y_train)
    scratch, _ = evaluate_model(model_scratch, X_valid, y_valid, threshold=threshold)
    sklearn, _ = evaluate_model(model_sklearn, X_valid, y_valid, threshold=threshold)
    return pd.DataFrame([scratch, sklearn], index=["from scratch", "scikit-learn"])


def save_model(model, preprocessor, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"model_{name}.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (output_dir / f"preprocessor_{name}.pkl").open("wb") as file:
        pickle.dump(preprocessor, file)


def load_model(output_dir, name):
    output_dir = Path(output_dir)
    with (output_dir / f"model_{name}.pkl").open("rb") as file:
        model = pickle.load(file)
    with (output_dir / f"preprocessor_{name}.pkl").open("rb") as file:
        preprocessor = pickle.load(file)
    return model, preprocessor


def submit_model(model, preprocessor, X_test, ids, output_dir, name):
    ids = pd.Series(ids).reset_index(drop=True)
    if len(ids) != len(X_test) or ids.isna().any() or not ids.is_unique:
        raise ValueError("ID test harus unik dan sesuai jumlah data test")
    ids = ids.astype(str)
    if not ids.is_unique:
        raise ValueError("ID test harus tetap unik setelah diubah menjadi teks")
    predictions = model.predict(preprocessor.transform(X_test))
    if len(predictions) != len(ids) or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Prediksi harus berjumlah sesuai data test dan bernilai 0 atau 1")
    submission = pd.DataFrame({"id": ids, "is_canceled": predictions})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / f"submission_{name}.csv", index=False)
    return submission
