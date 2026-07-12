import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, ax):

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(
        xx,
        yy,
        Z,
        levels=[Z.min(), 0, Z.max()],
        colors=["blue", "black", "red"],
        linestyles=["--", "-", "--"],
        alpha =0.3
    )
    
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k"
    )

    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Support Vectors"
    )

    ax.legend()
    accuracy = model.score(X,y)
    ax.set_title(
    f"Kernel: {model.kernel}\n"
    )

    if model.kernel == "linear":
        margin = 2 / np.linalg.norm(model.coef_)
        margin_text = f"{margin:.2f}"
    else:
        margin_text = "N/A"
    num_sv = len(model.support_vectors_)
    info = (
        f"Kernel: {model.kernel.upper()}\n"
        f"C: {model.C}\n"
        f"Support Vectors: {num_sv}\n"
        f"Accuracy: {accuracy * 100:.1f}%\n"
        f"Margin Width: {margin_text}"
    )
    ax.text(
    0.02,
    0.98,
    info,
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.8)
    )
    
    