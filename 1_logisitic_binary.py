import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import math


def predict_data(x, y, theta):
    y_pred = sigmoid(x.dot(theta))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(x, y, theta, alpha, iterations):
    prev_loss = 10e9
    loss = 10e8
    i = 0
    while(prev_loss-loss > 10e-4 and i < iterations):
        i += 1
        prev_loss = loss
        wtx = y.dot(1-sigmoid(x.dot(theta))) + (1-y).dot(sigmoid(x.dot(theta)))
        prediction = sigmoid(x.dot(theta))
        loss = (0.5/(y.size))*sum((y-prediction)**2)
        theta = theta - (alpha * sum(x.dot(wtx)))
    return theta


def perf_measure(y_actual, y_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i] == 1:
            TP += 1
        if y_predict[i] == 1 and y_actual[i] != y_predict[i]:
            FP += 1
        if y_actual[i] == y_predict[i] == 0:
            TN += 1
        if y_predict[i] == 0 and y_actual[i] != y_predict[i]:
            FN += 1
    return TP, FP, TN, FN


# Importing the dataset
dataset = pd.read_excel('data3.xlsx')

train = dataset[:-40]
test = dataset[-40:]

X_train = train.iloc[:, 0:4].values
X_train = np.insert(X_train, 0, np.ones(60), 1)
Y_train = train.iloc[:, 4].values - 1

X_test = test.iloc[:, 0:4].values
X_test = np.insert(X_test, 0, np.ones(40), 1)
Y_test = test.iloc[:, 4].values - 1

# Logistic Regression
np.random.seed(10)
theta = np.random.rand(5)
theta = logistic_regression(X_train, Y_train, theta, 0.000001, 1000)
Y_pred = predict_data(X_test, Y_test, theta)

print(Y_test)
print(Y_pred)

TP, FP, TN, FN = perf_measure(Y_test, Y_pred)

print("Logistic Regression")
print("Weights : ", end='')
print(theta)

print("Accuracy : ", end='')
print((TP+TN)/(TP+TN+FP+FN))
print("Sensitivity : ", end='')
print(TP/(TP+FN))
print("Specificity : ", end='')
print(TN/(TN+FP))

# Output
# Logistic Regression
# Weights : [ 0.65292853 -0.60771288  0.25025509  0.47295025  0.42944494]
# Accuracy : 0.95
# Sensitivity : 0.9354838709677419
# Specificity : 1.0
