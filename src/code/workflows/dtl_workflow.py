import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

from models.decision_tree import DecisionTreeID3
from models.pruning import prune_cost_complexity, prune_macro_f1, prune_reduced_error
from workflows.workflow import calibrate_threshold, evaluate_model, train_model
from preprocessing.preprocessing import create_preprocessor


def compare_pruning(X_train, y_train, X_valid, y_valid, max_depth=16, min_samples_split=10,
                    ccp_alpha=0.00035, seed=42):
    X_grow, X_prune, y_grow, y_prune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    pre = create_preprocessor(X_grow)
    grow, pruning, valid = pre.fit_transform(X_grow), pre.transform(X_prune), pre.transform(X_valid)

    model_rep = train_model(
        DecisionTreeID3(max_depth=max_depth, min_samples_split=min_samples_split), grow, y_grow
    )
    before, _ = evaluate_model(model_rep, valid, y_valid)
    accuracy_before = accuracy_score(y_prune, model_rep.predict(pruning))
    prune_reduced_error(model_rep, pruning, y_prune)
    assert accuracy_score(y_prune, model_rep.predict(pruning)) >= accuracy_before
    after_rep, _ = evaluate_model(model_rep, valid, y_valid)

    model_f1 = train_model(
        DecisionTreeID3(max_depth=max_depth, min_samples_split=min_samples_split), grow, y_grow
    )
    prune_macro_f1(model_f1, pruning, y_prune)
    after_f1, _ = evaluate_model(model_f1, valid, y_valid)

    pre_full = create_preprocessor(X_train)
    full_train = pre_full.fit_transform(X_train)
    full_valid = pre_full.transform(X_valid)
    model_ccp = DecisionTreeID3(max_depth=None, min_samples_split=min_samples_split).fit(full_train, y_train)
    prune_cost_complexity(model_ccp, ccp_alpha=ccp_alpha)
    after_ccp, _ = evaluate_model(model_ccp, full_valid, y_valid)

    return pd.DataFrame(
        [before, after_rep, after_f1, after_ccp],
        index=[
            "Sebelum pruning (80% grow)",
            "Reduced-Error Pruning (accuracy)",
            "Macro-F1 Pruning",
            f"Cost-Complexity Pruning (100% train, alpha={ccp_alpha})",
        ],
    )[["accuracy", "f1_kelas_0", "f1_kelas_1", "macro_f1"]]


def train_final_dtl(X_train, y_train, max_depth=None, min_samples_split=10,
                    ccp_alpha=0.00035, calibrate=True, seed=42):
    if ccp_alpha is not None:
        threshold = None
        if calibrate:
            X_it, X_ic, y_it, y_ic = train_test_split(
                X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
            )
            pre_cal = create_preprocessor(X_it)
            model_cal = DecisionTreeID3(
                max_depth=max_depth, min_samples_split=min_samples_split
            ).fit(pre_cal.fit_transform(X_it), y_it)
            prune_cost_complexity(model_cal, ccp_alpha=ccp_alpha)
            threshold, _ = calibrate_threshold(
                y_ic, model_cal.predict_proba(pre_cal.transform(X_ic))[:, 1],
                np.linspace(0.30, 0.60, 61),
            )
        preprocessor = create_preprocessor(X_train)
        X_mat = preprocessor.fit_transform(X_train)
        model = DecisionTreeID3(
            max_depth=max_depth, min_samples_split=min_samples_split, threshold=threshold
        ).fit(X_mat, y_train)
        prune_cost_complexity(model, ccp_alpha=ccp_alpha)
        return model, preprocessor

    X_grow, X_prune, y_grow, y_prune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    preprocessor = create_preprocessor(X_grow)
    grow = preprocessor.fit_transform(X_grow)
    pruning = preprocessor.transform(X_prune)
    model = DecisionTreeID3(max_depth=max_depth or 16, min_samples_split=min_samples_split).fit(grow, y_grow)
    prune_reduced_error(model, pruning, y_prune)
    return model, preprocessor
