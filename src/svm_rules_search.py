"""Explore single RBF-SVM parameters using only labeled train.csv.

Each split learns imputation, encoding, scaling and feature weights only from
its training partition. sklearn SVC is used for exploration, not submission.
The validation folds overlap earlier studies and are not unbiased holdouts.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    data = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    X = data.drop(columns=["id", "is_canceled"])
    y = data["is_canceled"].to_numpy()
    out = ROOT / "output/SVM/experiments/rules_parameter_search.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    for seed in (21, 42):
        settings = ((4, 0.16), (4, 0.12), (2, 0.20)) if seed == 21 else ((4, 0.16), (2, 0.20))
        if all(any(r["seed"] == seed and r["C"] == c and r["gamma"] == g for r in rows)
               for c, g in settings):
            continue
        train, valid = train_test_split(np.arange(len(y)), test_size=0.2,
                                        random_state=seed, stratify=y)
        prep = Pipeline([("prep", preprocessor(X.iloc[train])),
                         ("weights", MutualInformationWeights(exponent=0.75))])
        x_train = prep.fit_transform(X.iloc[train], y[train])
        x_valid = prep.transform(X.iloc[valid])
        for C, gamma in settings:
            if any(r["seed"] == seed and r["C"] == C and r["gamma"] == gamma for r in rows):
                continue
            start = time.perf_counter()
            model = SVC(C=C, gamma=gamma, tol=0.003, cache_size=256)
            model.fit(x_train, y[train])
            scores = model.decision_function(x_valid)
            row = {"seed": seed, "n_train": len(train), "n_valid": len(valid),
                   "C": C, "gamma": gamma, "mi_exponent": 0.75,
                   "seconds": time.perf_counter() - start}
            for threshold in (-0.3, -0.2, -0.1, 0.0, 0.1):
                row[f"f1_{threshold:.1f}"] = f1_score(
                    y[valid], scores >= threshold, average="macro")
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(row, flush=True)


if __name__ == "__main__":
    main()
