from sklearn.svm import SVC


def train_svm(X, y, kernel="rbf", C=1.0, gamma="scale",degree = 3):

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree
    )

    model.fit(X, y)

    return model