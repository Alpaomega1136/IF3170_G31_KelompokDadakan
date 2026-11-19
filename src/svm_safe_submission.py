"""One scratch SVM with separately implemented, training-only preprocessing.

Train and predict use NumPy, pandas and SciPy mathematical routines. The optional
comparison in validate() imports SVC only for a fair library baseline. Kernel
optimizer: Fan, Chen, Lin (2005), https://www.jmlr.org/papers/v6/fan05a.html.
Continuous mutual information: Ross (2014), https://doi.org/10.1371/journal.pone.0087357.
"""
import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy

from models.svm_working_set import WorkingSetSVM
from preprocessing.svm_safe_preprocessing import SafeSVMPreprocessor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "SVM" / "safe"
CONFIG = {"C": 4.0, "gamma": 0.16, "tol": 0.003, "max_iter": 200000,
          "cache_mb": 512, "seed": 2026, "threshold": -0.2,
          "mi_exponent": 0.75, "mi_estimator": "knn", "mi_neighbors": 5,
          "mi_seed": 42, "mi_min_score": 0.002}
BASELINE_THRESHOLD = -0.3


def load_data():
    train = pd.read_csv(ROOT / "data/train.csv", dtype={"id": str})
    test = pd.read_csv(ROOT / "data/test.csv", dtype={"id": str})
    sample = pd.read_csv(ROOT / "data/sample_submission.csv", dtype={"id": str})
    X = train.drop(columns=["id", "is_canceled"])
    X_test = test.drop(columns="id")
    y = train["is_canceled"].to_numpy()
    if X.columns.tolist() != X_test.columns.tolist() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Train/test feature schema or labels are unexpected")
    if not test["id"].is_unique or not sample["id"].equals(test["id"]):
        raise ValueError("Sample submission IDs do not match test IDs")
    return X, y, X_test, test, sample


def make_preprocessor():
    return SafeSVMPreprocessor(exponent=CONFIG["mi_exponent"],
                               min_score=CONFIG["mi_min_score"],
                               estimator=CONFIG["mi_estimator"],
                               n_neighbors=CONFIG["mi_neighbors"],
                               random_state=CONFIG["mi_seed"])


def make_model():
    return WorkingSetSVM(C=CONFIG["C"], gamma=CONFIG["gamma"], kernel="rbf",
                         tol=CONFIG["tol"], max_iter=CONFIG["max_iter"],
                         cache_mb=CONFIG["cache_mb"], threshold=CONFIG["threshold"])


def validate(X, y):
    # scikit-learn is only used here for splitting, metrics, and comparison.
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    fit, valid = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=CONFIG["seed"], stratify=y)
    prep = make_preprocessor()
    X_fit = prep.fit_transform(X.iloc[fit], y[fit])
    X_valid = prep.transform(X.iloc[valid])
    rows = []
    for name in ("scratch", "sklearn_reference"):
        model = (make_model() if name == "scratch" else
                 SVC(C=CONFIG["C"], gamma=CONFIG["gamma"], tol=CONFIG["tol"],
                     cache_size=CONFIG["cache_mb"]))
        start = time.perf_counter()
        model.fit(X_fit, y[fit])
        elapsed = time.perf_counter() - start
        if name == "scratch" and not model.converged_:
            raise RuntimeError("Scratch SVM did not converge during validation")
        scores = model.decision_function(X_valid)
        prediction = scores >= CONFIG["threshold"]
        row = {"model": name, "seed": CONFIG["seed"], "n_train": len(fit),
               "n_valid": len(valid), "threshold": CONFIG["threshold"],
               "macro_f1": f1_score(y[valid], prediction, average="macro"),
               "macro_f1_threshold_minus_0_3": f1_score(y[valid], scores >= BASELINE_THRESHOLD, average="macro"),
               "accuracy": accuracy_score(y[valid], prediction), "fit_seconds": elapsed,
               "kkt_gap": model.kkt_gap_ if name == "scratch" else None}
        rows.append(row)
        print(f"{name}: macro F1={row['macro_f1']:.5f}, train={elapsed:.1f}s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "validation.csv", index=False)
    return results


def make_submission(X, y, X_test, test, sample):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prep = make_preprocessor()
    X_train = prep.fit_transform(X, y)
    X_test_ready = prep.transform(X_test)
    model = make_model()
    start = time.perf_counter()
    model.fit(X_train, y)
    elapsed = time.perf_counter() - start
    if not model.converged_:
        raise RuntimeError("Final scratch SVM did not converge; submission not written")
    scores = model.decision_function(X_test_ready)
    predictions = (scores >= CONFIG["threshold"]).astype(int)
    baseline_predictions = (scores >= BASELINE_THRESHOLD).astype(int)
    model_path, prep_path = OUTPUT / "svm_scratch.pkl", OUTPUT / "preprocessor.pkl"
    model.save(model_path)
    with prep_path.open("wb") as file:
        pickle.dump(prep, file)
    with prep_path.open("rb") as file:
        restored_prep = pickle.load(file)
    restored_model = WorkingSetSVM.load(model_path)
    restored_scores = restored_model.decision_function(restored_prep.transform(X_test))
    np.testing.assert_array_equal(predictions, (restored_scores >= CONFIG["threshold"]).astype(int))
    np.testing.assert_array_equal(baseline_predictions, (restored_scores >= BASELINE_THRESHOLD).astype(int))
    result = pd.DataFrame({"id": test["id"], "is_canceled": predictions})
    if (result.columns.tolist() != sample.columns.tolist() or len(result) != len(test)
            or not result["id"].equals(sample["id"]) or result.isna().any().any()
            or not result["is_canceled"].isin([0, 1]).all()):
        raise ValueError("Invalid submission columns, IDs or predictions")
    path = OUTPUT / "submission_svm_safe.csv"
    result.to_csv(path, index=False)
    baseline_path = OUTPUT / "submission_svm_safe_baseline.csv"
    pd.DataFrame({"id": test["id"], "is_canceled": baseline_predictions}).to_csv(baseline_path, index=False)
    metadata = {"config": CONFIG, "training_rows": len(y), "test_rows": len(test),
                "fit_seconds": elapsed, "n_iter": model.n_iter_, "kkt_gap": model.kkt_gap_,
                "n_support": model.n_support_, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "baseline_threshold": BASELINE_THRESHOLD,
                "baseline_submission_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                "predictions_changed_from_baseline": int(np.sum(predictions != baseline_predictions)),
                "inputs_sha256": {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                                  for name in ('train.csv', 'test.csv', 'sample_submission.csv')}}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Final SVM: {elapsed:.1f}s; KKT gap={model.kkt_gap_:.6f}; {path}", flush=True)
    return result, metadata
