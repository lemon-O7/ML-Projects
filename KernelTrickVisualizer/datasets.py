from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

def add_outlier(X, point=(1.5, 1.5)):
    X = X.copy()
    X[0] = point
    return X


def generate_dataset(dataset_type):

    if dataset_type == "circles":
        X, y = make_circles(
            n_samples=300,
            noise=0.1,
            factor=0.4,
            random_state=42
        )

    elif dataset_type == "moons":
        X, y = make_moons(
            n_samples=300,
            noise=0.1,
            random_state=42
        )

    elif dataset_type == "blobs":
        X, y = make_blobs(
            n_samples=300,
            centers=2,
            cluster_std=2.0,
            random_state=42
        )
    
    elif dataset_type == "linear":
        X, y = make_blobs(
            n_samples =  300,
            center = [(-2,-2),(2,2)],
            cluster_std=0.8,
            random_state=42
        )

    else:
        raise ValueError("Invalid dataset")

    return X, y