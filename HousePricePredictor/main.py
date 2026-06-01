import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def cost_function(X,y,weights,bias) :
    predictions = np.dot(X, weights) + bias

    cost = (1/ (2* len(y))) * np.sum((predictions - y) ** 2)

    return cost

def gradient_descent(X,y,weights,bias,learning_rate) :
    predictions = np.dot(X,weights) + bias

    dw = (1/len(y)) * np.dot(X.T, (predictions - y))

    db = (1/len(y)) * np.sum(predictions - y)

    weights = weights - learning_rate * dw

    bias = bias - learning_rate * db

    return weights, bias

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.csv")

data = pd.read_csv(file_path)

X = data[["Area", "Bedrooms", "Bathrooms", "Age"]].values
y = data["Price"].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)

y = (y - y_mean) / y_std

weights = np.zeros(X.shape[1])
bias = 0

predictions = np.dot(X, weights) + bias

cost = cost_function(X,y,weights,bias)

learning_rate = 0.01

epochs = 1000

costs = []

for i in range(epochs):

    weights, bias = gradient_descent(
        X,
        y,
        weights,
        bias,
        learning_rate
    )

    cost = cost_function(
        X,
        y,
        weights,
        bias
    )

    costs.append(cost)

new_house = np.array([1700,3,2,10])

new_house_normalized = (new_house - X_mean) / X_std

prediction = (
    np.dot(new_house_normalized, weights)
    + bias
)

prediction = (
    prediction * y_std
) + y_mean

print("Predicted House Price:")

print(prediction)