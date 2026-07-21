
DATASETS = [
    "linear",
    "blobs",
    "moons",
    "circles",
    "XOR"
]

DEFAULT_DATASET = "circles"

KERNEL_CONFIGS = [{"kernel" : "linear", "C": 100},
                   {"kernel" : "rbf", "C":100},
                   {"kernel" : "poly", "C": 100, "degree" : 3}]