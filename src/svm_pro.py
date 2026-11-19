"""Reproducible scratch SVM candidate with booking-feature preprocessing.

The model is a single NumPy working-set SVM. SVC is used only as a reference
on the labeled validation split. No test labels or external data are used.
"""
import hashlib
import json
import os
import pickle
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import svm_optimized as common
from experiment_svm import preprocessor as basic_preprocessor
from models.svm_working_set import WorkingSetSVM
from preprocessing.svm_engineered import BookingFeatures
from preprocessing.svm_feature_weights import MutualInformationWeights

ROOT = common.ROOT
OUTPUT = ROOT / "output" / "SVM" / "pro"
CONFIG = {"C": 4.0, "gamma": 0.16, "tol": 0.003, "max_iter": 200000,
          "cache_mb": 512, "seed": 2026, "threshold": -0.3,
          "dates": False, "mi_exponent": 0.75, "mi_seed": 42}


def build_preprocessor(X):
    feature_builder = BookingFeatures(dates=CONFIG["dates"], durations=False, party=False)
    # This transform is row-wise; fitting here only records column names.
    columns = feature_builder.fit_transform(X)
    return Pipeline([
        ("features", feature_builder),
        ("preprocessing", basic_preprocessor(columns)),
        ("feature_weights", MutualInformationWeights(exponent=CONFIG["mi_exponent"],
                                                     min_score=0.002,
                                                     random_state=CONFIG["mi_seed"])),
    ])


def make_model():
    return WorkingSetSVM(C=CONFIG["C"], gamma=CONFIG["gamma"], kernel="rbf",
                         tol=CONFIG["tol"], max_iter=CONFIG["max_iter"],
                         cache_mb=CONFIG["cache_mb"], threshold=CONFIG["threshold"])


def validate(X, y):
    fit, valid = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=CONFIG["seed"], stratify=y)
    prep = build_preprocessor(X.iloc[fit])
    X_fit = prep.fit_transform(X.iloc[fit], y[fit])
    X_valid = prep.transform(X.iloc[valid])
    rows = []
    for name in ("scratch", "sklearn"):
        model = (make_model() if name == "scratch" else
                 SVC(C=CONFIG["C"], gamma=CONFIG["gamma"], tol=CONFIG["tol"],
                     cache_size=CONFIG["cache_mb"]))
        start = time.perf_counter()
        model.fit(X_fit, y[fit])
        elapsed = time.perf_counter() - start
        if name == "scratch" and not model.converged_:
            raise RuntimeError("Scratch SVM did not converge during validation")
        scores = model.decision_function(X_valid)
        predicted = scores >= CONFIG["threshold"]
        row = {"model": name, "seed": CONFIG["seed"], "n_train": len(fit),
               "n_valid": len(valid), "threshold": CONFIG["threshold"],
               "macro_f1": f1_score(y[valid], predicted, average="macro"),
               "macro_f1_threshold_0": f1_score(y[valid], scores >= 0, average="macro"),
               "accuracy": accuracy_score(y[valid], predicted), "fit_seconds": elapsed,
               "kkt_gap": model.kkt_gap_ if name == "scratch" else None}
        rows.append(row)
        print(f"{name}: macro F1={row['macro_f1']:.5f}, train={elapsed:.1f}s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "validation.csv", index=False)
    return results


def make_submission(X, y, X_test, test, sample):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prep = build_preprocessor(X)
    X_train = prep.fit_transform(X, y)
    X_test_ready = prep.transform(X_test)
    model = make_model()
    start = time.perf_counter()
    model.fit(X_train, y)
    elapsed = time.perf_counter() - start
    if not model.converged_:
        raise RuntimeError("Final scratch SVM did not converge; no submission written")
    predictions = model.predict(X_test_ready).astype(int)
    model_path = OUTPUT / "svm_scratch.pkl"
    prep_path = OUTPUT / "preprocessor.pkl"
    model.save(model_path)
    with prep_path.open("wb") as file:
        pickle.dump(prep, file)
    with prep_path.open("rb") as file:
        reloaded_prep = pickle.load(file)
    reloaded = WorkingSetSVM.load(model_path)
    np.testing.assert_array_equal(predictions, reloaded.predict(reloaded_prep.transform(X_test)))
    submission = pd.DataFrame({"id": test["id"], "is_canceled": predictions})
    if (submission.columns.tolist() != sample.columns.tolist() or len(submission) != len(test)
            or not submission["id"].equals(sample["id"]) or submission.isna().any().any()
            or not submission["is_canceled"].isin([0, 1]).all()):
        raise ValueError("Submission does not match the sample schema and test IDs")
    path = OUTPUT / "submission_svm_pro.csv"
    submission.to_csv(path, index=False)
    metadata = {"config": CONFIG, "training_rows": len(y), "test_rows": len(test),
                "fit_seconds": elapsed, "n_iter": model.n_iter_, "kkt_gap": model.kkt_gap_,
                "n_support": model.n_support_, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Final model: {elapsed:.1f}s, KKT gap={model.kkt_gap_:.6f}; {path}", flush=True)
    return submission, metadata
