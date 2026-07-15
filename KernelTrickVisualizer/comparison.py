from dataset_generator import generate_dataset,add_outlier
from svm_utils import train_svm
from visualizations import plot_decision_boundary
import matplotlib.pyplot as plt

def compare_kernels(X,y,configs,axes) :
    

    models = []


    for config in configs :
        model = train_svm(
            X,
            y,
            **config
        )
        models.append(model)


    for model, ax in zip(models, axes):
        plot_decision_boundary (
            model,
            X,
            y,
            ax
        )
   