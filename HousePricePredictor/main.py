import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def cost_function(x,y,m,b):
    predictions = m*x + b

    cost = (1/(2*len(x))) * np.sum((predictions-y)**2)

    return cost

def gradient_descent(x,y,m,b,learning_rate):
    predictions = m*x+b

    dm = (1/len(x)) * np.sum((predictions-y) * x)
    db = (1/len(x)) * np.sum(predictions-y) 

    m = m-learning_rate * dm
    b = b-learning_rate*db

    return m,b

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.csv")

data = pd.read_csv(file_path)

print(data.columns)

x = data["Area"].values
y = data["Price"].values

plt.scatter(x,y)
plt.xlabel("Area")
plt.ylabel("Prices")
plt.title("House Prices")

x_original = x.copy()
y_original = y.copy()

x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
costs = []
for i in range(epochs):

    m, b = gradient_descent(x, y, m, b, learning_rate)
    cost = cost_function(x, y, m, b)
    costs.append(cost)
    if i % 100 == 0:
        print(f"Epoch {i}: Cost = {cost}")

print("Final m:", m)
print("Final b:", b)



plt.plot(costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")

pred = m*x+b

# Sort values for smooth regression line
sorted_indices = np.argsort(x)

x_sorted = x[sorted_indices]
pred_sorted = pred[sorted_indices]

# Regression graph
plt.figure()

plt.scatter(x, y)
plt.plot(x_sorted, pred_sorted)

plt.xlabel("Normalized Area")
plt.ylabel("Normalized Price")

plt.title("Linear Regression")

# Cost graph
plt.figure()

plt.plot(costs)

plt.xlabel("Epoch")
plt.ylabel("Cost")

plt.title("Cost Reduction Over Time")

plt.show()