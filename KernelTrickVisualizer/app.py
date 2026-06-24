from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

def train_svm(X, y, kernel, C, gamma):
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma
    )

    model.fit(X, y)

    return model