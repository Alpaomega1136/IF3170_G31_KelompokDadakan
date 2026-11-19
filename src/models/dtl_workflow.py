from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

from models.decision_tree import DecisionTreeID3
from models.pruning import prune_reduced_error
from models.workflow import evaluate_model, train_model
from preprocessing.preprocessing import create_preprocessor


def compare_pruning(X_train, y_train, X_valid, y_valid, max_depth=13, seed=42):
    X_grow, X_prune, y_grow, y_prune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    pre = create_preprocessor(X_grow)
    grow, pruning, valid = pre.fit_transform(X_grow), pre.transform(X_prune), pre.transform(X_valid)
    model = train_model(DecisionTreeID3(max_depth=max_depth), grow, y_grow)
    before, _ = evaluate_model(model, valid, y_valid)
    accuracy_before = accuracy_score(y_prune, model.predict(pruning))
    prune_reduced_error(model, pruning, y_prune)
    assert accuracy_score(y_prune, model.predict(pruning)) >= accuracy_before
    after, _ = evaluate_model(model, valid, y_valid)
    return pd.DataFrame([before, after], index=["Sebelum pruning", "Sesudah pruning"])[["accuracy", "macro_f1"]]


def train_final_dtl(X_train, y_train, max_depth=16, min_samples_split=10, seed=42):
    X_grow, X_prune, y_grow, y_prune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    preprocessor = create_preprocessor(X_grow)
    grow = preprocessor.fit_transform(X_grow)
    pruning = preprocessor.transform(X_prune)
    model = DecisionTreeID3(max_depth=max_depth, min_samples_split=min_samples_split).fit(grow, y_grow)
    prune_reduced_error(model, pruning, y_prune)
    return model, preprocessor


