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
    # Pangkas simpul tanpa menurunkan macro F1 pada data pruning.
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


def prune_cost_complexity(model, ccp_alpha):
    # Referensi: Breiman dkk. (1984), Classification and Regression Trees.
    if model.tree_ is None:
        raise ValueError("Fit the model before pruning")
    if ccp_alpha < 0:
        raise ValueError("ccp_alpha must be non-negative")
    if ccp_alpha == 0 or "counts" not in model.tree_:
        return model

    total_n = float(model.tree_["counts"].sum())
    if total_n <= 0:
        return model

    def node_risk(node):
        counts = node["counts"]
        return (float(counts.sum()) / total_n) * float(model._entropy(counts))

    # Simpan risiko dan jumlah daun, termasuk data yang tetap di simpul induk.
    def build_stats(node):
        r = node_risk(node)
        if "children" not in node:
            node["_ccp"] = (r, r, 1)
            return r, 1
        sr, leaves = 0.0, 0
        child_counts = np.zeros_like(node["counts"])
        for child in node["children"].values():
            child_counts = child_counts + child["counts"]
            cr, cl = build_stats(child)
            sr += cr
            leaves += cl
        residual = node["counts"] - child_counts
        res_n = float(residual.sum())
        if res_n > 0:
            sr += (res_n / total_n) * float(model._entropy(residual))
            leaves += 1
        node["_ccp"] = (r, sr, leaves)
        return sr, leaves

    def find_weakest(node, path, best):
        if "children" not in node:
            return best
        r, sr, leaves = node["_ccp"]
        if leaves > 1:
            alpha = (r - sr) / (leaves - 1)
            if best is None or alpha < best[0]:
                best = (alpha, node, list(path))
        for child in node["children"].values():
            path.append(child)
            best = find_weakest(child, path, best)
            path.pop()
        return best

    build_stats(model.tree_)
    while True:
        weakest = find_weakest(model.tree_, [model.tree_], None)
        if weakest is None or weakest[0] > ccp_alpha:
            break
        _, target, path = weakest
        r_t, sr_t, leaves_t = target["_ccp"]
        delta_risk = r_t - sr_t
        delta_leaves = 1 - leaves_t
        target.pop("feature")
        target.pop("threshold")
        target.pop("children")
        target["_ccp"] = (r_t, r_t, 1)
        for ancestor in path[:-1]:
            ra, sra, la = ancestor["_ccp"]
            ancestor["_ccp"] = (ra, sra + delta_risk, la + delta_leaves)

    def cleanup(node):
        node.pop("_ccp", None)
        for child in node.get("children", {}).values():
            cleanup(child)

    cleanup(model.tree_)
    return model


if __name__ == "__main__":
    from models.decision_tree import DecisionTreeID3

    # Uji cost-complexity pruning untuk data yang tertahan di simpul induk.
    X_ccp = np.array([["A"], ["A"], ["B"], ["B"], ["C"], ["D"]], dtype=object)
    y_ccp = np.array([0, 0, 1, 1, 0, 1])
    m1 = prune_cost_complexity(DecisionTreeID3(max_depth=3, min_samples_leaf=2).fit(X_ccp, y_ccp), 0.8)
    assert "children" not in m1.tree_
    m2 = prune_cost_complexity(DecisionTreeID3(max_depth=3, min_samples_leaf=2).fit(X_ccp, y_ccp), 0.2)
    assert "children" in m2.tree_
    print("Uji pruning (termasuk residual risk CCP) lolos")
