import numpy as np
from sklearn.datasets import make_blobs


def generate_dataset(
    n_samples=200,
    centers=2,
    cluster_std=1.2,
    random_state=42
):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    return X, y