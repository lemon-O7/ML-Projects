import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].set_title("Linear")
axes[1].set_title("RBF")
axes[2].set_title("Polynomial")

plt.show()