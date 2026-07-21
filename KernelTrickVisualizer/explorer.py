import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from plotting import compare_kernels
from datasets import generate_dataset
from config import DEFAULT_DATASET, DATASETS, KERNEL_CONFIGS

def run_explorer():
    
    X,y = generate_dataset(DEFAULT_DATASET)

    def redraw() :
        for ax in axes:
            ax.clear()
        compare_kernels(
            X,
            y,
            KERNEL_CONFIGS,
            axes
        )

        fig.canvas.draw_idle()

    def update_dataset(label) :
        nonlocal X,y
        X,y = generate_dataset(label)
        redraw()

    def update_parameters(val):

        c = c_slider.val
        gamma = gamma_slider.val

        for config in KERNEL_CONFIGS:

            config["C"] = c

            if config["kernel"] != "linear":
                config["gamma"] = gamma

        redraw()

    fig, axes = plt.subplots(1, len(KERNEL_CONFIGS), figsize=(6*len(KERNEL_CONFIGS), 6))
    fig.subplots_adjust(bottom=0.35)
    fig.suptitle("SVM Explorer")
    slider_ax = plt.axes([0.2, 0.08, 0.6, 0.03])
    # create slider...

    c_slider = Slider(
        ax=slider_ax,
        label="C",
        valmin=1,
        valmax=100,
        valinit=10,
        valstep=1
    )

    c_slider.on_changed(update_parameters)

    radio_ax = fig.add_axes([0.9,0.0,0.1,0.14])

    dataset_radio = RadioButtons(
        radio_ax,
        DATASETS
    )

    dataset_radio.on_clicked(update_dataset)

    gamma_ax = fig.add_axes([0.2,0.04, 0.6, 0.03])

    gamma_slider = Slider(
        ax=gamma_ax,
        label="Gamma",
        valmin=0.01,
        valmax=5,
        valinit=1,
    )

    gamma_slider.on_changed(update_parameters)

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.show()