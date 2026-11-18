"""Train and validate a scratch RBF SVM and generate a Kaggle submission.

Run from the project root: python src/svm_optimized.py
This performs two validation splits and trains on all labeled data. Training
can take several minutes and the kernel-row cache can use up to 512 MiB.
"""
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_svm import preprocessor as basic_preprocessor
from models.svm_working_set import WorkingSetSVM

BASE_OUTPUT = ROOT / "output" / "SVM" / "optimized"
DEFAULT_CONFIG = {"C": 4.0, "gamma": 0.16, "kernel": "rbf", "tol": 0.003,
                  "max_iter": 200000, "cache_mb": 512, "seeds": [21, 42]}
OUTPUT = BASE_OUTPUT
CONFIG = DEFAULT_CONFIG.copy()
preprocessor = basic_preprocessor


def configure_baseline():
    """Restore the unweighted experiment when switching notebooks in a kernel."""
    global OUTPUT, CONFIG, preprocessor
    OUTPUT = BASE_OUTPUT
    CONFIG = DEFAULT_CONFIG.copy()
    preprocessor = basic_preprocessor


def make_model():
    return WorkingSetSVM(C=CONFIG["C"], gamma=CONFIG["gamma"], kernel=CONFIG["kernel"],
                         tol=CONFIG["tol"], max_iter=CONFIG["max_iter"],
                         cache_mb=CONFIG["cache_mb"])


def load_data():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    test = pd.read_csv(ROOT / "data" / "test.csv", dtype={"id": str})
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype={"id": str})
    X = train.drop(columns=["id", "is_canceled"])
    X_test = test.drop(columns="id")
    y = train["is_canceled"].to_numpy()
    if X.columns.tolist() != X_test.columns.tolist() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Train/test feature schema or labels are unexpected")
    if not test["id"].is_unique or not sample["id"].equals(test["id"]):
        raise ValueError("Sample submission IDs do not match test IDs")
    return X, y, X_test, test, sample


def calibrate_threshold(y, scores):
    thresholds = np.arange(-0.5, 0.251, 0.025)
    values = np.array([f1_score(y, scores >= threshold, average="macro") for threshold in thresholds])
    # For ties, use the threshold nearest zero.
    choices = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=0))
    return float(thresholds[choices[np.argmin(np.abs(thresholds[choices]))]])


def save_validation_plot(results, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    table = results.pivot(index="seed", columns="model", values="macro_f1")
    fig = Figure(figsize=(7, 4))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    positions = np.arange(len(table))
    width = 0.8 / len(table.columns)
    for index, column in enumerate(table.columns):
        ax.bar(positions + (index - (len(table.columns) - 1) / 2) * width,
               table[column].to_numpy(), width=width, label=column)
    ax.set_xticks(positions, labels=table.index.astype(str))
    ax.set_ylim(0, 1)
    ax.set(xlabel="Seed pembagian data", ylabel="Macro F1 validasi",
           title="SVM scratch dan scikit-learn")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output / "validation_macro_f1.png", dpi=150)


def validate(X, y):
    records = []
    for seed in CONFIG["seeds"]:
        pool, holdout = train_test_split(np.arange(len(y)), test_size=0.2,
                                        random_state=seed, stratify=y)
        fit_idx, tune_idx = train_test_split(pool, test_size=0.2,
                                             random_state=seed, stratify=y[pool])
        prep = preprocessor(X.iloc[fit_idx])
        X_fit = prep.fit_transform(X.iloc[fit_idx], y[fit_idx])
        X_tune = prep.transform(X.iloc[tune_idx])
        X_holdout = prep.transform(X.iloc[holdout])
        for kind in ("scratch", "sklearn"):
            model = (make_model() if kind == "scratch" else
                     SVC(C=CONFIG["C"], gamma=CONFIG["gamma"], kernel="rbf",
                         tol=CONFIG["tol"], cache_size=CONFIG["cache_mb"]))
            start = time.perf_counter()
            model.fit(X_fit, y[fit_idx])
            seconds = time.perf_counter() - start
            if kind == "scratch" and not model.converged_:
                raise RuntimeError(f"Scratch SVM did not converge on seed {seed}")
            tune_scores = model.decision_function(X_tune)
            threshold = calibrate_threshold(y[tune_idx], tune_scores)
            scores = model.decision_function(X_holdout)
            predictions = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[holdout], predictions, labels=[0, 1]).ravel()
            record = {"seed": seed, "model": kind, "n_train": len(fit_idx),
                      "n_calibration": len(tune_idx), "n_validation": len(holdout),
                      "C": CONFIG["C"], "gamma": CONFIG["gamma"], "threshold": threshold,
                      "macro_f1": f1_score(y[holdout], predictions, average="macro"),
                      "macro_f1_threshold_0": f1_score(y[holdout], scores >= 0, average="macro"),
                      "accuracy": accuracy_score(y[holdout], predictions),
                      "tn": tn, "fp": fp, "fn": fn, "tp": tp, "fit_seconds": seconds,
                      "n_support": model.n_support_ if kind == "scratch" else int(model.n_support_.sum()),
                      "n_iter": model.n_iter_ if kind == "scratch" else int(model.n_iter_.sum()),
                      "kkt_gap": model.kkt_gap_ if kind == "scratch" else None}
            records.append(record)
            print(f"seed={seed} {kind}: Macro F1={record['macro_f1']:.5f}, "
                  f"threshold={threshold:.3f}, training={seconds:.1f}s", flush=True)
    results = pd.DataFrame(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "validation.csv", index=False)
    save_validation_plot(results, OUTPUT)
    return results


def make_submission(X, y, X_test, test, sample, results):
    scratch_results = results[results["model"] == "scratch"]
    if len(scratch_results) != len(CONFIG["seeds"]):
        raise ValueError("Missing scratch validation results")
    threshold = float(scratch_results["threshold"].median())
    prep = preprocessor(X)
    X_train = prep.fit_transform(X, y)
    X_test_ready = prep.transform(X_test)
    model = make_model()
    model.threshold = threshold
    start = time.perf_counter()
    model.fit(X_train, y)
    fit_seconds = time.perf_counter() - start
    if not model.converged_:
        raise RuntimeError("Final SVM did not converge; submission not written")
    predictions = model.predict(X_test_ready).astype(int)
    model_path = OUTPUT / "svm_rbf_scratch.pkl"
    prep_path = OUTPUT / "preprocessor.pkl"
    model.save(model_path)
    with prep_path.open("wb") as file:
        pickle.dump(prep, file)
    with prep_path.open("rb") as file:
        loaded_prep = pickle.load(file)
    loaded_model = WorkingSetSVM.load(model_path)
    np.testing.assert_array_equal(predictions, loaded_model.predict(loaded_prep.transform(X_test)))
    submission = pd.DataFrame({"id": test["id"], "is_canceled": predictions})
    if (submission.columns.tolist() != sample.columns.tolist() or len(submission) != len(test)
            or not submission["id"].equals(sample["id"]) or submission.isna().any().any()
            or not submission["is_canceled"].isin([0, 1]).all()):
        raise ValueError("Invalid submission schema, ID, or predictions")
    submission.to_csv(OUTPUT / "submission_svm_optimized.csv", index=False)
    metadata = {"config": CONFIG, "threshold": threshold, "training_rows": len(y),
                "test_rows": len(test), "fit_seconds": fit_seconds, "n_iter": model.n_iter_,
                "kkt_gap": model.kkt_gap_, "n_support": model.n_support_,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Final scratch SVM trained on {len(y)} rows in {fit_seconds:.1f}s; "
          f"KKT gap={model.kkt_gap_:.6f}, threshold={threshold:.3f}", flush=True)
    print(f"Submission: {OUTPUT / 'submission_svm_optimized.csv'}", flush=True)
    return submission, metadata


def make_zero_threshold_submission(X_test, test, sample):
    """Save an additional candidate using the standard SVM threshold 0.

    The zero threshold was inspected on the two reported validation splits, so
    its reported validation performance also reflects model selection.
    """
    with (OUTPUT / "preprocessor.pkl").open("rb") as file:
        prep = pickle.load(file)
    model = WorkingSetSVM.load(OUTPUT / "svm_rbf_scratch.pkl")
    model.threshold = 0.0
    predictions = model.predict(prep.transform(X_test)).astype(int)
    submission = pd.DataFrame({"id": test["id"], "is_canceled": predictions})
    if (submission.columns.tolist() != sample.columns.tolist() or len(submission) != len(test)
            or not submission["id"].equals(sample["id"]) or submission.isna().any().any()
            or not submission["is_canceled"].isin([0, 1]).all()):
        raise ValueError("Invalid zero-threshold submission")
    model.save(OUTPUT / "svm_rbf_scratch_threshold0.pkl")
    np.testing.assert_array_equal(
        predictions, WorkingSetSVM.load(OUTPUT / "svm_rbf_scratch_threshold0.pkl")
        .predict(prep.transform(X_test)))
    path = OUTPUT / "submission_svm_threshold0.csv"
    submission.to_csv(path, index=False)
    print(f"Zero-threshold submission: {path}", flush=True)
    return submission


def main():
    X, y, X_test, test, sample = load_data()
    results = validate(X, y)
    make_submission(X, y, X_test, test, sample, results)
    make_zero_threshold_submission(X_test, test, sample)


if __name__ == "__main__":
    main()
