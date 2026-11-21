import numpy as np
from sklearn.model_selection import train_test_split

from models.decision_tree import DecisionTreeID3
from models.pruning import prune_cost_complexity
from preprocessing.preprocessing import create_preprocessor
from workflows.workflow import calibrate_threshold


def train_dtl(X_train, y_train, max_depth=None, min_samples_split=10,
              ccp_alpha=0.00035, calibrate=True, seed=42):
    threshold = None
    if calibrate:
        X_fit, X_calibration, y_fit, y_calibration = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
        )
        preprocessor_calibration = create_preprocessor(X_fit)
        model_calibration = DecisionTreeID3(
            max_depth=max_depth, min_samples_split=min_samples_split
        ).fit(preprocessor_calibration.fit_transform(X_fit), y_fit)
        if ccp_alpha is not None:
            prune_cost_complexity(model_calibration, ccp_alpha)
        threshold, _ = calibrate_threshold(
            y_calibration,
            model_calibration.predict_proba(preprocessor_calibration.transform(X_calibration))[:, 1],
            np.linspace(0.30, 0.60, 61),
        )

    preprocessor = create_preprocessor(X_train)
    model = DecisionTreeID3(
        max_depth=max_depth, min_samples_split=min_samples_split, threshold=threshold
    ).fit(preprocessor.fit_transform(X_train), y_train)
    if ccp_alpha is not None:
        prune_cost_complexity(model, ccp_alpha)
    return model, preprocessor
