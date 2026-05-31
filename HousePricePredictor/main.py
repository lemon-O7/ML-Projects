import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def cost_function(X,y,weights,bias) :
    predictions = np.dot(X, weights) + bias

    cost = (1/ (2* len(y))) * np.sum((predictions - y) ** 2)

    return cost

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.csv")

data = pd.read_csv(file_path)

X = data[["Area", "Bedrooms", "Bathrooms", "Age"]].values
y = data["Price"].values

weights = np.zeros(X.shape[1])
bias = 0

predictions = np.dot(X, weights) + bias


