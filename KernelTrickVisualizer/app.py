from dataset_generator import generate_dataset
from comparison import compare_kernels

DATASET = "circles"
X , y = generate_dataset(DATASET)
KERNEL_CONFIGS = [{"kernel" : "linear", "C": 100},
                   {"kernel" : "rbf", "C":100},
                   {"kernel" : "poly", "C": 100, "degree" : 3}]

compare_kernels(X, y, KERNEL_CONFIGS)