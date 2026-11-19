"""Compare candidate booking feature transforms on labeled train data only.

SVC is a fast proxy for model selection. Final training must use scratch SVM.
"""
import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--baseline-params-only", action="store_true")
    parser.add_argument("--mi-exponent", type=float, default=.5)
    args = parser.parse_args()
    df = pd.read_csv(ROOT / "data/train.csv")
    X = df.drop(columns=["id", "is_canceled"])
    y = df.is_canceled.to_numpy()
    train, valid = train_test_split(np.arange(len(y)), test_size=.2, random_state=args.seed, stratify=y)
    if args.n_train < len(train):
        train, _ = train_test_split(train, train_size=args.n_train, random_state=args.seed, stratify=y[train])
    options = {
        "baseline": dict(dates=False, durations=False, party=False),
        "missing": dict(dates=False, durations=False, party=False, missing=True),
        "missing_special": dict(dates=False, durations=False, party=False, missing=True,
                                missing_columns=("special_requests",)),
        "dates": dict(dates=True, durations=False, party=False),
        "duration_party": dict(dates=False, durations=True, party=True),
        "duration": dict(dates=False, durations=True, party=False),
        "party": dict(dates=False, durations=False, party=True),
        "log": dict(dates=False, durations=False, party=False, log=True),
        "all": dict(dates=True, durations=True, party=True, missing=True, log=True),
    }
    records = []
    for name, opts in options.items():
        if args.features and name not in args.features:
            continue
        t = time.perf_counter()
        engineer = BookingFeatures(**opts)
        train_df = engineer.fit_transform(X.iloc[train])
        valid_df = engineer.transform(X.iloc[valid])
        transform = Pipeline([("prep", preprocessor(train_df)),
                              ("weight", MutualInformationWeights(exponent=args.mi_exponent))])
        Xtr = transform.fit_transform(train_df, y[train])
        Xval = transform.transform(valid_df)
        params = [(4, .16)] if args.baseline_params_only else [(4, .16), (8, .16), (4, .08)]
        for C, gamma in params:
            model = SVC(C=C, gamma=gamma, cache_size=384)
            model.fit(Xtr, y[train])
            decision = model.decision_function(Xval)
            score = f1_score(y[valid], decision >= 0, average="macro")
            score_minus_point_three = f1_score(y[valid], decision >= -.3, average="macro")
            row = dict(seed=args.seed, n_train=len(train), feature_set=name, C=C, gamma=gamma,
                       mi_exponent=args.mi_exponent,
                       macro_f1=score, macro_f1_minus_point_three=score_minus_point_three,
                       elapsed_seconds=time.perf_counter()-t)
            records.append(row)
            print(row, flush=True)
    out = ROOT / "output/SVM/experiments"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / f"feature_engineering_{args.seed}_{args.n_train}_mi{args.mi_exponent}.csv", index=False)


if __name__ == "__main__":
    main()
