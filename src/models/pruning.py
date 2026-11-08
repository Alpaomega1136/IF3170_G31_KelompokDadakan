import numpy as np


def prune_reduced_error(model, X, y):
    """Prune bottom-up when a leaf is at least as accurate on pruning data."""
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
