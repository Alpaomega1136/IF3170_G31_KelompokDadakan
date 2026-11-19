"""Compare reproducible preprocessing candidates on labeled data only.

SVC is used as a fast comparison for the NumPy implementation; the final
submission must be produced by WorkingSetSVM. Scores are selection results,
not an untouched estimate: the training subsets overlap between splits.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from experiment_svm import ROOT, preprocessor
from preprocessing.svm_engineered import BookingFeatures
from preprocessing.svm_feature_weights import MutualInformationWeights


def build_preprocessor(X, dates, exponent):
    features = BookingFeatures(dates=dates, durations=False, party=False)
    engineered = features.fit_transform(X)
    return Pipeline([
        ("features", features),
        ("preprocess", preprocessor(engineered)),
        ("weights", MutualInformationWeights(exponent=exponent)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 77])
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data/train.csv", dtype={"id": str})
    X = data.drop(columns=["id", "is_canceled"])
    y = data.is_canceled.to_numpy()
    rows = []
    for seed in args.seeds:
        train, valid = train_test_split(np.arange(len(y)), test_size=0.2,
                                        random_state=seed, stratify=y)
        for dates, exponent in [(False, 0.5), (True, 0.5), (False, 0.75), (True, 0.75)]:
            prep = build_preprocessor(X.iloc[train], dates, exponent)
            x_train = prep.fit_transform(X.iloc[train], y[train])
            x_valid = prep.transform(X.iloc[valid])
            start = time.perf_counter()
            model = SVC(C=4, gamma=0.16, tol=0.003, cache_size=384).fit(x_train, y[train])
            scores = model.decision_function(x_valid)
            row = dict(seed=seed, n_train=len(train), dates=dates, exponent=exponent,
                       C=4, gamma=0.16,
                       f1_threshold_0=f1_score(y[valid], scores >= 0, average="macro"),
                       f1_threshold_minus_03=f1_score(y[valid], scores >= -0.3, average="macro"),
                       fit_seconds=time.perf_counter() - start)
            rows.append(row)
            print(row, flush=True)
            out = ROOT / "output/SVM/experiments/candidate_study.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
