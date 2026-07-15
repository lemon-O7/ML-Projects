import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from comparison import compare_kernels

def launch_explorer(X,y,configs):
    
    def update(val) :
        c_val = c_slider.val
        for config in configs:
            config["C"] = c_val
        
        for ax in axes :
            ax.clear()
        
        compare_kernels(
            X,
            y,
            configs,
            axes
        )
        fig.canvas.draw_idle()

    fig, axes = plt.subplots(1, len(configs), figsize=(6*len(configs), 6))
    fig.subplots_adjust(bottom=0.20)
    fig.suptitle("SVM Explorer")
    slider_ax = plt.axes([0.2, 0.08, 0.6, 0.03])
    # create slider...

    c_slider = Slider(
        ax=slider_ax,
        label="C",
        valmin=0.01,
        valmax=100,
        valinit=100
    )
    c_slider.on_changed(update)
    compare_kernels(
        X,
        y,
        configs,
        axes
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.show()