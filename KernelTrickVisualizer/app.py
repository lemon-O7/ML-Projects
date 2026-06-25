from dataset_generator import generate_dataset
from svm_utils import train_svm
from visualizations import plot_decision_boundary


X, y = generate_dataset("circles")

model = train_svm(
    X,
    y,
    kernel="linear"
)

plot_decision_boundary(
    model,
    X,
    y
)