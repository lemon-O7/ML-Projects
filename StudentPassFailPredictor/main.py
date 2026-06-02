import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.csv")

df = pd.read_csv(file_path)

X = df[["Hours_Studied", "Attendance"]].values
y = df["Passed"].values