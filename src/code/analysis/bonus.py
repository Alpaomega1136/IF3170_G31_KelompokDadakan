from itertools import count
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from models.logistic_regression import LogisticRegression
from models.svm import WorkingSetSVM

ACCENT_COLOR = "#963f24"


def plot_dtl_tree(model, feature_names, top_n=7):
    if model.tree_ is None or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Pohon harus sudah dilatih dan top_n harus bilangan bulat positif.")

    # Simpul diperluas per tingkat agar aturan dari akar terlihat lebih dahulu.
    nodes = [model.tree_]
    branches = {}
    for index, node in enumerate(nodes):
        if len(branches) == top_n:
            break
        if node.get("children"):
            first = len(nodes)
            nodes.extend(node["children"].values())
            branches[index] = list(zip(node["children"], range(first, len(nodes))))

    positions = {}
    leaf_x = count()

    def place(index, depth):
        children = [place(child, depth + 1) for _, child in branches.get(index, [])]
        x = (children[0] + children[-1]) / 2 if children else next(leaf_x)
        positions[index] = (x, -depth)
        return x

    place(0, 0)
    depth = max(-y for _, y in positions.values())
    width = max(x for x, _ in positions.values()) + 1
    fig, ax = plt.subplots(figsize=(max(8, 2.3 * width), max(4, 1.65 * (depth + 1))), layout="constrained")
    branch_color = "#2f6f5e"
    for parent, children in branches.items():
        x0, y0 = positions[parent]
        for key, child in children:
            x1, y1 = positions[child]
            ax.plot([x0, x1], [y0, y1], color=branch_color, linewidth=1.25, zorder=1)
            label = {"left": "≤", "right": ">", None: "kosong"}.get(key, str(key))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center",
                    fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    for i, node in enumerate(nodes):
        counts = ", ".join(f"{c}: {n}" for c, n in zip(model.classes_, node["counts"]))
        if i in branches:
            feature = str(feature_names[node["feature"]]).split("__", 1)[-1]
            threshold = node["threshold"]
            title = feature if threshold is None else f"{feature} ≤ {threshold:.3g}"
        elif node.get("children"):
            title = "Cabang lanjutan"
        else:
            title = f"Prediksi {node['prediction']}"
        ax.text(*positions[i], f"{fill(title, width=24)}\n{counts}", ha="center", va="center",
                fontsize=9, zorder=3, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef8f2", edgecolor=branch_color))
    ax.set(xlim=(-0.55, width - 0.45), ylim=(-depth - 0.45, 0.45),
           title=f"Pohon ID3 setelah pruning | {len(branches)} simpul percabangan")
    ax.axis("off")
    return fig


def save_lr_loss_video(X, y, path, random_state=42):
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Pembuatan video MP4 memerlukan FFmpeg pada PATH.")
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Visualisasi Logistic Regression memerlukan tepat satu fitur.")
    model = LogisticRegression(lr=0.1, n_epoch=50, batch_size=64, tol=0, random_state=random_state)
    model.fit(X, y, record_trajectory=True)
    path_weights = np.asarray(model.trajectory_)
    axis = [np.linspace(w.min() - 0.3, w.max() + 0.3, 60) for w in path_weights.T]
    theta0, theta1 = np.meshgrid(*axis)
    z = theta0[..., None] + theta1[..., None] * X[:, 0]
    log_loss = (np.logaddexp(0, z) - y * z).mean(axis=-1)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    contours = ax.contour(theta0, theta1, log_loss, levels=14, cmap="viridis")
    fig.colorbar(contours, ax=ax, pad=0.035, label="Log-loss")
    ax.set(xlabel=r"$\theta_0$ (intersep)", ylabel=r"$\theta_1$ (koefisien fitur)",
           title="Lintasan Logistic Regression pada kontur log-loss")
    ax.scatter(*path_weights[0], marker="x", s=75, color="black", label="Inisialisasi", zorder=4)
    line, = ax.plot([], [], color=ACCENT_COLOR, linewidth=2, label="Lintasan pelatihan", zorder=5)
    point, = ax.plot([], [], "o", color=ACCENT_COLOR, markersize=6, zorder=6)
    message = ax.text(0.03, 0.04, "", transform=ax.transAxes,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ax.legend(loc="upper right")

    try:
        writer = FFMpegWriter(fps=8, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(fig, str(path), dpi=120):
            for epoch, (weights, loss) in enumerate(zip(path_weights, [np.log(2), *model.loss_history_])):
                line.set_data(*path_weights[:epoch + 1].T)
                point.set_data([weights[0]], [weights[1]])
                message.set_text(f"Epoch {epoch:02d} | Log-loss {loss:.4f}")
                writer.grab_frame()
            for _ in range(8):
                writer.grab_frame()
    finally:
        plt.close(fig)
    return model


def plot_svm_kernels(X, y, feature_names):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Visualisasi SVM memerlukan tepat dua fitur.")

    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = np.maximum(0.12 * (hi - lo), 0.2)
    grid_x, grid_y = np.meshgrid(*[np.linspace(a, b, 160) for a, b in zip(lo - pad, hi + pad)])
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    kernels = [("Linear", "linear", {}), ("Polynomial", "polynomial", {"degree": 2}),
               ("Gaussian / RBF", "rbf", {"gamma": 0.16})]
    colors = ("#25608a", ACCENT_COLOR)
    fig, axes = plt.subplots(1, len(kernels), figsize=(17, 5.6), sharex=True, sharey=True, layout="constrained")
    for ax, (label, kernel, options) in zip(axes, kernels):
        model = WorkingSetSVM(C=4, kernel=kernel, tol=0.005, cache_mb=64, **options).fit(X, y)
        if not model.converged_:
            raise RuntimeError(f"SVM kernel {label} belum konvergen untuk visualisasi.")
        values = model.decision_function(grid).reshape(grid_x.shape)
        region_levels = [min(-1, values.min()), 0, max(1, values.max())]
        ax.contourf(grid_x, grid_y, values, levels=region_levels, colors=colors, alpha=0.12)
        contours = ax.contour(grid_x, grid_y, values, levels=[-1, 0, 1],
                              colors="#20242b", linestyles=["--", "-", "--"], linewidths=[1, 1.7, 1])
        ax.clabel(contours, fmt={-1: "−1", 0: "0", 1: "+1"}, fontsize=8)
        for target, color in zip(model.classes_, colors):
            ax.scatter(*X[y == target].T, color=color, s=12, alpha=0.54)
        ax.scatter(*model.support_vectors_.T, s=30, linewidths=0.7, facecolors="none", edgecolors="#20242b")
        ax.set(title=f"{label} | {model.n_support_} support vector", xlabel=feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=f"Label aktual {i}")
               for i, c in enumerate(colors)]
    handles += [Line2D([], [], color="#20242b", linestyle="none", marker="o", markerfacecolor="none", label="Support vector"),
                Line2D([], [], color="#20242b", label="Batas f = 0"),
                Line2D([], [], color="#20242b", linestyle="--", label="Margin f = ±1")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Batas keputusan dan margin SVM pada dua fitur")
    return fig


def plot_naive_bayes_graph(feature_names, target="is_canceled"):
    features = list(feature_names)
    middle = (len(features) + 1) // 2
    fig, ax = plt.subplots(figsize=(13, 8), layout="constrained")
    for x, names in [(-1, features[:middle]), (1, features[middle:])]:
        for y, feature in zip(np.linspace(0.9, -0.9, len(names)), names):
            ax.annotate("", xy=(x * 0.76, y), xytext=(x * 0.16, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT_COLOR, alpha=0.56, lw=1.2))
            ax.add_patch(FancyBboxPatch((x - 0.24, y - 0.037), 0.48, 0.074,
                                       boxstyle="round,pad=0.006", facecolor="#fff4ed", edgecolor=ACCENT_COLOR))
            ax.text(x, y, feature, ha="center", va="center", fontsize=10, family="monospace")
    ax.text(0, 0, target, ha="center", va="center", color="white", fontsize=12, family="monospace",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=ACCENT_COLOR, edgecolor=ACCENT_COLOR))
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.05, 1.05), title="Struktur Naive Bayes | Kelas → fitur")
    ax.axis("off")
    return fig
