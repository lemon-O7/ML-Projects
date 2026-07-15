from dataset_generator import generate_dataset
from interactive import launch_explorer


DATASET = "circles"
X , y = generate_dataset(DATASET)
KERNEL_CONFIGS = [{"kernel" : "linear", "C": 100},
                   {"kernel" : "rbf", "C":100},
                   {"kernel" : "poly", "C": 100, "degree" : 3}]

launch_explorer(X,y,KERNEL_CONFIGS)
