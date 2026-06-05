import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "spam_dataset.csv")

df = pd.read_csv(file_path)

encoder = LabelEncoder()

y = encoder.fit_transform(df["label"])

print(y)