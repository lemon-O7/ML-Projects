from dataset_generator import generate_dataset,add_outlier
from svm_utils import train_svm
from visualizations import plot_decision_boundary
import matplotlib.pyplot as plt


X , y = generate_dataset("circles")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X_modified = add_outlier(X)

linear = train_svm(
    X,
    y,
    kernel="linear",
    C=1
)
rbf = train_svm(
    X,
    y,
    kernel="rbf",
    C=1
)
polynomial = train_svm(
    X,
    y,
    kernel="poly",
    C=1
)

plot_decision_boundary(
    linear,
    X,
    y,
    axes[0]
)
plot_decision_boundary(
    rbf,
    X,
    y,
    axes[1]
)
plot_decision_boundary(
    polynomial,
    X,
    y,
    axes[2]
)

plt.tight_layout()
plt.show()