import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from plotting import compare_kernels
from datasets import generate_dataset
from config import (
    DEFAULT_DATASET,
    DEFAULT_C,
    DEFAULT_GAMMA,
    DATASETS,
    KERNEL_CONFIGS
)

def run_explorer():
    
    X,y = generate_dataset(DEFAULT_DATASET)
    current_dataset = DEFAULT_DATASET
    
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
        nonlocal X,y,current_dataset
        
        for button in button_map.values():
            button.color = "0.85"

        button_map[label].color = "lightgreen"

        fig.canvas.draw_idle()

        current_dataset = label
        X,y = generate_dataset(label)
        redraw()
        fig.canvas.draw_idle()

    def update_parameters(val):

        c = c_slider.val
        gamma = gamma_slider.val

        for config in KERNEL_CONFIGS:

            config["C"] = c

            if config["kernel"] != "linear":
                config["gamma"] = gamma

        redraw()

    def reset(event):
        nonlocal X, y, current_dataset

        current_dataset = DEFAULT_DATASET
        X, y = generate_dataset(DEFAULT_DATASET)

        c_slider.set_val(DEFAULT_C)
        gamma_slider.set_val(DEFAULT_GAMMA)

        for button in button_map.values():
            button.color = "0.85"

        button_map[DEFAULT_DATASET].color = "lightgreen"

        redraw()


    fig, axes = plt.subplots(1, len(KERNEL_CONFIGS), figsize=(6*len(KERNEL_CONFIGS), 6))
    fig.subplots_adjust(bottom=0.35)
    fig.suptitle("SVM Explorer")
    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    # create slider...

    c_slider = Slider(
        ax=slider_ax,
        label="C",
        valmin=1,
        valmax=100,
        valinit=DEFAULT_C,
        valstep=1
    )

    c_slider.on_changed(update_parameters)

    

    gamma_ax = fig.add_axes([0.2,0.04, 0.6, 0.03])

    gamma_slider = Slider(
        ax=gamma_ax,
        label="Gamma",
        valmin=0.01,
        valmax=5,
        valinit=DEFAULT_GAMMA,
    )

    gamma_slider.on_changed(update_parameters)

    button_height = 0.05
    start_x = 0.18
    button_y = 0.15
    button_width = 0.12
    gap = 0.01
    
    buttons = []
    button_map = {}

    for i, dataset in enumerate(DATASETS):

        ax = fig.add_axes([
            start_x + i * (button_width + gap),
            button_y,
            button_width,
            button_height
        ])

        btn = Button(
            ax,
            dataset,
            color="0.85",
            hovercolor="0.75"
        )
        buttons.append(btn)
        button_map[dataset] = btn 

    for button, dataset in zip(buttons, DATASETS):

        button.on_clicked(
            lambda event, d=dataset: update_dataset(d)
        )
    button_map[DEFAULT_DATASET].ax.set_facecolor("lightgreen")

    reset_ax = fig.add_axes([0.84, 0.04, 0.10, 0.07])

    reset_button = Button(
        reset_ax,
        "Reset",
        color="0.85",
        hovercolor="0.75"
    )

    reset_button.on_clicked(reset)

    redraw()
    plt.show()