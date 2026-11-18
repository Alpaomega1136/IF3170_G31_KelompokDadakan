"""Evaluate the chosen weighted SVM and fixed thresholds on another split."""
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import svm_weighted


def main():
    svm_weighted.configure()
    svm = svm_weighted.svm
    X, y, _, _, _ = svm.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=100
    )
    prep = svm.preprocessor(X_train)
    fit = prep.fit_transform(X_train, y_train)
    valid = prep.transform(X_valid)
    model = svm.make_model()
    start = time.perf_counter()
    model.fit(fit, y_train)
    elapsed = time.perf_counter() - start
    if not model.converged_:
        raise RuntimeError("SVM did not converge on additional validation split")
    scores = model.decision_function(valid)
    rows = []
    for threshold in (0.0, -0.3):
        prediction = scores >= threshold
        row = {"seed": 100, "n_train": len(y_train), "n_validation": len(y_valid),
               "threshold": threshold, "macro_f1": f1_score(y_valid, prediction, average="macro"),
               "accuracy": accuracy_score(y_valid, prediction), "fit_seconds": elapsed,
               "kkt_gap": model.kkt_gap_}
        rows.append(row)
        print(row, flush=True)
    svm.OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(svm.OUTPUT / "validation_seed100.csv", index=False)


if __name__ == "__main__":
    main()
