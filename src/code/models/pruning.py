import numpy as np


def prune_reduced_error(model, X, y):
    if model.tree_ is None:
        raise ValueError("Fit the model before pruning")
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != model.n_features_in_ or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("Pruning data must match the model and contain labels")

    def prune(node, rows, labels):
        if "children" not in node or not len(labels):
            return
        feature = node["feature"]
        threshold = node["threshold"]
        keys = []
        for value in rows[:, feature]:
            if model._missing(value):
                keys.append(None)
            elif threshold is None:
                keys.append(value)
            else:
                keys.append("left" if value <= threshold else "right")
        for key, child in node["children"].items():
            mask = np.array([item == key for item in keys], dtype=bool)
            prune(child, rows[mask], labels[mask])

        root = model.tree_
        model.tree_ = node
        try:
            correct_subtree = np.count_nonzero(model.predict(rows) == labels)
        finally:
            model.tree_ = root
        correct_leaf = np.count_nonzero(labels == node["prediction"])
        if correct_leaf >= correct_subtree:
            node.pop("feature")
            node.pop("threshold")
            node.pop("children")

    prune(model.tree_, X, y)
    return model


def prune_macro_f1(model, X, y):
    """Prune bottom-up without decreasing macro F1 on the pruning set."""
    if model.tree_ is None:
        raise ValueError("Fit the model before pruning")
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != model.n_features_in_ or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("Pruning data must match the model and contain labels")
    if not np.isin(y, [0, 1]).all() or not np.isin(model.classes_, [0, 1]).all():
        raise ValueError("Macro F1 pruning requires binary labels 0 and 1")

    def score(pred):
        values = []
        for label in (0, 1):
            true_positive = np.count_nonzero((y == label) & (pred == label))
            denominator = np.count_nonzero(y == label) + np.count_nonzero(pred == label)
            values.append(2 * true_positive / denominator if denominator else 0.0)
        return sum(values) / 2

    predictions = model.predict(X)

    def prune(node, indices):
        nonlocal predictions
        if "children" not in node or len(indices) == 0:
            return
        feature, threshold = node["feature"], node["threshold"]
        keys = []
        for value in X[indices, feature]:
            if model._missing(value):
                keys.append(None)
            elif threshold is None:
                keys.append(value)
            else:
                keys.append("left" if value <= threshold else "right")
        for key, child in node["children"].items():
            mask = np.array([item == key for item in keys], dtype=bool)
            prune(child, indices[mask])
        candidate = predictions.copy()
        candidate[indices] = node["prediction"]
        if score(candidate) >= score(predictions):
            node.pop("feature")
            node.pop("threshold")
            node.pop("children")
            predictions = candidate

    prune(model.tree_, np.arange(len(y)))
    return model
