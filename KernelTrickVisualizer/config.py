
DATASETS = [
    "linear",
    "blobs",
    "moons",
    "circles",
    "XOR"
]

DATASET_EXPLANATIONS = {
    "linear": "Linearly separable dataset. All kernels perform similarly.",
    "blobs": "Simple clustered dataset with two Gaussian blobs.",
    "moons": "Two interleaving moons requiring a nonlinear boundary.",
    "circles": "Concentric circles that demonstrate the kernel trick.",
    "XOR": "Classic XOR pattern that is not linearly separable."
}

DEFAULT_C = 10
DEFAULT_GAMMA = 1

DEFAULT_DATASET = "circles"

KERNEL_CONFIGS = [{"kernel" : "linear", "C": 100},
                   {"kernel" : "rbf", "C":100},
                   {"kernel" : "poly", "C": 100, "degree" : 3}]