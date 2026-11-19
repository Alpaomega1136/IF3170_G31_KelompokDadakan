"""Compare a fixed train-only selection candidate to Pro on a new split.

Use only after selecting C, gamma and threshold on other splits. This split
shares examples with previous training sets, so it is not an external test.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from experiment_svm import ROOT, preprocessor
from preprocessing.svm_feature_weights import MutualInformationWeights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[303])
    parser.add_argument("--output", default="rules_parameter_holdout.csv")
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data/train.csv", dtype={"id": str})
    X = data.drop(columns=["id", "is_canceled"])
    y = data["is_canceled"].to_numpy()
    rows = []
    for seed in args.seeds:
        train, valid = train_test_split(np.arange(len(y)), test_size=0.2,
                                        random_state=seed, stratify=y)
        prep = Pipeline([("prep", preprocessor(X.iloc[train])),
                         ("weights", MutualInformationWeights(exponent=0.75))])
        x_train = prep.fit_transform(X.iloc[train], y[train])
        x_valid = prep.transform(X.iloc[valid])
        score_cache = {}
        for name, C, gamma, threshold in (("pro", 4., 0.16, -0.3),
                                          ("candidate", args.C, args.gamma, args.threshold)):
            start = time.perf_counter()
            if (C, gamma) not in score_cache:
                score_cache[(C, gamma)] = SVC(C=C, gamma=gamma, tol=0.003, cache_size=256).fit(
                    x_train, y[train]).decision_function(x_valid)
            scores = score_cache[(C, gamma)]
            row = {"name": name, "seed": seed, "n_train": len(train), "n_valid": len(valid),
                   "C": C, "gamma": gamma, "threshold": threshold,
                   "f1": f1_score(y[valid], scores >= threshold, average="macro"),
                   "seconds": time.perf_counter() - start}
            rows.append(row)
            print(row, flush=True)
    out = ROOT / "output/SVM/experiments" / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
