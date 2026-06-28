from dataset_generator import generate_dataset,add_outlier
from svm_utils import train_svm
from visualizations import plot_decision_boundary



X , y = generate_dataset("circles")

X_modified = add_outlier(X)

model = train_svm(
    X,
    y,
    kernel="linear",
    C=100
)



plot_decision_boundary(
    model,
    X,
    y
)