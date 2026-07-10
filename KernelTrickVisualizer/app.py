from dataset_generator import generate_dataset,add_outlier
from svm_utils import train_svm
from visualizations import plot_decision_boundary
import matplotlib.pyplot as plt

DATASET = "circles"
KERNELS = ["linear","rbf","poly"]

C = 100
GAMMA = "scale"
DEGREE = 3



X , y = generate_dataset(DATASET)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X_modified = add_outlier(X)

models = []

for kernel in KERNELS :
    model = train_svm(
        X,
        y,
        kernel=kernel,
        C=C
    )
    models.append(model)


for model, ax in zip(models, axes):
    plot_decision_boundary (
        model,
        X,
        y,
        ax
    )
plt.tight_layout()
plt.show()