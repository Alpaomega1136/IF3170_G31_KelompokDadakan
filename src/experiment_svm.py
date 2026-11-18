"""Reproducible SVM hyperparameter exploration; uses labeled train data only.

SVC is used as a quick reference for choosing promising configurations. Final
submission must use the independently implemented WorkingSetSVM.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.svm_working_set import WorkingSetSVM


def preprocessor(X, indicators=False):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=indicators)),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def prepare(seed, n_train, indicators=False):
    data = pd.read_csv(ROOT / "data/train.csv", dtype={"id": str})
    X = data.drop(columns=["id", "is_canceled"])
    y = data["is_canceled"].to_numpy()
    pool, valid = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    if n_train < len(pool):
        pool, _ = train_test_split(pool, train_size=n_train, random_state=seed, stratify=y[pool])
    prep = preprocessor(X.iloc[pool], indicators)
    return prep.fit_transform(X.iloc[pool]), y[pool], prep.transform(X.iloc[valid]), y[valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indicators", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()
    X, y, Xv, yv = prepare(args.seed, args.n_train, args.indicators)
    print(f"training={len(y)} valid={len(yv)} features={X.shape[1]} seed={args.seed}", flush=True)
    records = []
    for C in ((2, 4, 8) if args.focused else (1, 4, 16, 64)):
        for gamma in ((0.08, 0.16, 0.24) if args.focused else (0.01, 0.04, 0.16, 0.64)):
            start = time.perf_counter()
            model = (WorkingSetSVM(C=C, gamma=gamma, cache_mb=384, tol=0.002, max_iter=200000)
                     if args.scratch else SVC(C=C, gamma=gamma, cache_size=384))
            model.fit(X, y)
            fit_seconds = time.perf_counter() - start
            scores = model.decision_function(Xv)
            zero = f1_score(yv, scores >= 0, average="macro")
            thresholds = np.linspace(-0.8, 0.4, 25)
            metrics = [f1_score(yv, scores >= t, average="macro") for t in thresholds]
            winner = int(np.argmax(metrics))
            result = {"seed": args.seed, "n_train": len(y), "indicators": args.indicators,
                      "model": "scratch" if args.scratch else "sklearn", "C": C, "gamma": gamma,
                      "macro_f1_threshold_0": zero, "best_threshold": thresholds[winner],
                      "best_macro_f1": metrics[winner], "fit_seconds": fit_seconds,
                      "converged": getattr(model, "converged_", None),
                      "kkt_gap": getattr(model, "kkt_gap_", None)}
            records.append(result)
            print(result, flush=True)
    out = ROOT / "output/SVM/experiments"
    out.mkdir(parents=True, exist_ok=True)
    name = f"grid_{'focused_' if args.focused else ''}{'scratch' if args.scratch else 'sklearn'}_{args.seed}_{args.n_train}_{int(args.indicators)}.csv"
    pd.DataFrame(records).to_csv(out / name, index=False)
    print("Saved", out / name, flush=True)


if __name__ == "__main__":
    main()
