import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y):

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
    print(Z.min(), Z.max())
    plt.contour(
        xx,
        yy,
        Z,
        levels=[-1, 0, 1],
        colors=["blue", "black", "red"],
        linestyles=["--", "-", "--"]
    )
    
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k"
    )

    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Support Vectors"
    )

    plt.legend()

    plt.title(f"SVM Kernel = {model.kernel}")


    num_sv = len(model.support_vectors_)

    plt.text(
        0.02,
        0.98,
        f"Support Vectors: {num_sv}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.show()