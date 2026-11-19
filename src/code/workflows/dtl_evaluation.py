from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from models.decision_tree import DecisionTreeID3
from models.pruning import prune_macro_f1, prune_reduced_error
from preprocessing.preprocessing import create_preprocessor


def _validate_data(X, y):
    if not isinstance(X, pd.DataFrame) or len(X) != len(y) or len(X) == 0:
        raise ValueError("X must be a nonempty DataFrame with one label per row")
    if {"id", "is_canceled"} & set(X.columns):
        raise ValueError("Remove id and is_canceled from the feature DataFrame")
    if not np.isin(np.asarray(y), [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")


def _split_and_prepare(X_train, y_train, X_valid, seed):
    X_grow, X_prune, y_grow, y_prune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    preprocessor = create_preprocessor(X_grow)
    return (
        preprocessor.fit_transform(X_grow),
        preprocessor.transform(X_prune),
        preprocessor.transform(X_valid),
        y_grow,
        y_prune,
    )


def _scores(y_true, prediction):
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "f1_kelas_0": f1_score(y_true, prediction, pos_label=0, zero_division=0),
        "f1_kelas_1": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, prediction, average="macro", labels=[0, 1], zero_division=0),
    }


def tune_dtl(X, y, depths=(14, 15, 16), min_splits=(2, 10),
             min_leaves=(1, 5), n_splits=3, seed=42):
    """Evaluate ID3 settings and pruning on stratified folds.

    Pass raw features without ID/target as a DataFrame and binary labels as a
    Series or array. Each fold fits its preprocessing and pruning on training
    data only. Returns per-fold results and a summary sorted by mean macro F1.
    """
    _validate_data(X, y)
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds.split(X, y), 1):
        grow, pruning, valid, y_grow, y_prune = _split_and_prepare(
            X.iloc[train_index], y[train_index], X.iloc[valid_index], seed
        )
        for depth, split, leaf in product(depths, min_splits, min_leaves):
            scratch = DecisionTreeID3(
                max_depth=depth, min_samples_split=split, min_samples_leaf=leaf
            ).fit(grow, y_grow)
            models = [
                ("tanpa pruning", scratch),
                ("pruning accuracy", prune_reduced_error(deepcopy(scratch), pruning, y_prune)),
                ("pruning macro F1", prune_macro_f1(deepcopy(scratch), pruning, y_prune)),
            ]
            for name, model in models:
                rows.append({
                    "fold": fold, "max_depth": depth, "min_samples_split": split,
                    "min_samples_leaf": leaf, "pruning": name,
                    **_scores(y[valid_index], model.predict(valid)),
                })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["max_depth", "min_samples_split", "min_samples_leaf", "pruning"],
        dropna=False,
    ).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
    ).sort_values("macro_f1_mean", ascending=False)
    return detail, summary
