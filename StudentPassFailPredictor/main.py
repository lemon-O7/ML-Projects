import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def sigmoid(z):
    # core of logistic regression
    return 1 / (1 + np.exp(-z))

def predict_probability(X, weights, bias):
    # this is the theta transpose * x
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def cost_function(X, y, weights, bias) :
    # binary cross entropy function

    m = len(y)
    predictions = predict_probability(X, weights, bias)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = (-1/m) * np.sum(
        y * np.log(predictions) +
        (1-y) * np.log(1-predictions)
    )

    return cost
 
# gradient descent for update rule

def gradient_descent(X, y, weights, bias, learning_rate):
    m = len(y)

    predictions = predict_probability(X, weights, bias)

    dw = (1 / m) * np.dot(X.T, (predictions - y))
    db = (1 / m) * np.sum(predictions - y)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.csv")

df = pd.read_csv(file_path)

X = df[["Hours_Studied", "Attendance"]].values
y = df["Passed"].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

weights = np.zeros(X.shape[1])
bias = 0

#alpha
learning_rate = 0.1
epochs = 1000 #number of times we will train the model on entire dataset

costs = [] #tells us about error/accuracy in the model

for epoch in range(epochs):

    weights , bias = gradient_descent(X,y,weights,bias,learning_rate)

    cost = cost_function(X,y,weights,bias)
    costs.append(cost)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {cost}")
    
probabilities = predict_probability(X, weights, bias)

predictions = [1 if p>0.5 else 0 for p in probabilities]

print(predictions)

accuracy = np.mean(predictions == y)

print(f"Accuracy: {accuracy * 100:.2f}%")

plt.plot(costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training Loss")
plt.show()

# plot data points

plt.figure(figsize=(8,6))

for i in range(len(y)):
    
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1], marker='o')
    else:
        plt.scatter(X[i,0], X[i,1], marker='x')

# decision boundary

x_values = np.array([X[:,0].min(), X[:,0].max()])

y_values = -(weights[0] * x_values + bias) / weights[1]

plt.plot(x_values, y_values)

plt.xlabel("Hours Studied (scaled)")
plt.ylabel("Attendance (scaled)")
plt.title("Decision Boundary")

plt.show()