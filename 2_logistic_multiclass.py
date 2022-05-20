import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import math
from sklearn import preprocessing


def predict_data(x, y, theta1, theta2, theta3):
    y_pred1 = sigmoid(x.dot(theta1))
    y_pred1 = np.where(y_pred1 > 0.5, 1, 0)
    # print(y_pred1)
    y_pred2 = sigmoid(x.dot(theta2))
    y_pred2 = np.where(y_pred2 > 0.5, 2, 0)
    # print(y_pred2)
    y_pred3 = sigmoid(x.dot(theta3))
    y_pred3 = np.where(y_pred3 > 0.5, 3, 0)
    # print(y_pred3)
    y_pred = np.maximum(y_pred1, y_pred2)
    y_pred = np.maximum(y_pred, y_pred3)
    y_pred[y_pred == 0] = 1
    return y_pred


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(x, y, theta, alpha, iterations):
    prev_loss = 10e9
    loss = 10e8
    i = 0
    while(i < iterations):
        i += 1
        prev_loss = loss
        wtx = y.dot(1-sigmoid(x.dot(theta))) + (1-y).dot(sigmoid(x.dot(theta)))
        theta = theta - (alpha * sum(x.dot(wtx)))

        prediction = sigmoid(x.dot(theta))
        prediction = np.where(prediction > 0.5, 1, 0)
        loss = (0.5/(y.size))*sum((y-prediction)**2)
        # print(loss)
        if(loss > prev_loss+0.1):
            break
    return theta


def perf_measure(y_actual, y_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i] == 1:
            TP += 1
        elif y_predict[i] == 1 and y_actual[i] != y_predict[i]:
            FP += 1
        elif y_actual[i] == y_predict[i] != 1:
            TN += 1
        elif y_predict[i] != 1 and y_actual[i] != y_predict[i]:
            FN += 1
    return TP, FP, TN, FN


# Importing the dataset
dataset = pd.read_excel('data4.xlsx')
dataset = dataset.sample(frac=1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(dataset)
dataset = pd.DataFrame(np_scaled)

train = dataset[:-60]
test = dataset[-60:]

X_train = train.iloc[:, 0:4].values
X_train = np.insert(X_train, 0, np.ones(90), 1)
Y_train = train.iloc[:, 4].values

Y_train_1 = np.array(Y_train)
Y_train_1[Y_train_1 != 1] = 0
Y_train_1[Y_train_1 == 1] = 1

Y_train_2 = np.array(Y_train)
Y_train_2[Y_train != 2] = 0
Y_train_2[Y_train == 2] = 1

Y_train_3 = np.array(Y_train)
Y_train_3[Y_train == 3] = 1
Y_train_3[Y_train != 3] = 0
Y_train_3[Y_train == 3] = 1

X_test = test.iloc[:, 0:4].values
X_test = np.insert(X_test, 0, np.ones(60), 1)
Y_test = test.iloc[:, 4].values

# Logistic Regression one vs all
print("Logistic Regression one vs all")
np.random.seed(10)
theta1 = np.random.rand(5)
theta1 = logistic_regression(X_train, Y_train_1, theta1, 0.0000001, 200)
print("Weights 1 vs rest: ", end='')
print(theta1)

np.random.seed(10)
theta2 = np.random.rand(5)
theta2 = logistic_regression(X_train, Y_train_2, theta2, 0.0000001, 200)
print("Weights 2 vs rest: ", end='')
print(theta2)

np.random.seed(10)
theta3 = np.random.rand(5)
theta3 = logistic_regression(X_train, Y_train_3, theta3, 0.0000001, 200)
print("Weights 3 vs rest: ", end='')
print(theta3)

Y_pred = predict_data(X_test, Y_test, theta1, theta2, theta3)

print(Y_test)
print(Y_pred)

TP, FP, TN, FN = perf_measure(Y_test, Y_pred)
print("Overall Accuracy : ", end='')
print((TP+TN)/(TP+TN+FP+FN))

# Output
# Logistic Regression one vs all
# Weights 1 vs rest: [ 0.66167583 -0.62164485  0.30020616  0.33848861  0.36778603]
# Weights 2 vs rest: [ 0.67563197 -0.53987734  0.34264836  0.39071561  0.38442485]
# Weights 3 vs rest: [ 0.67324437 -0.55386607  0.33538738  0.38178066  0.38157829]
# [1 3 2 3 2 3 3 2 3 3 1 3 1 1 3 1 2 1 1 1 1 1 3 1 3 1 2 3 1 2 3 2 2 3 2 1 3
#  2 1 1 2 2 3 1 2 2 3 1 1 3 2 2 3 3 3 2 3 1 3 1]
# [1 3 3 3 3 3 3 3 3 3 1 3 1 1 3 1 3 1 1 1 1 1 3 1 3 1 3 3 1 3 3 3 3 3 3 1 3
#  3 1 1 3 3 3 1 3 3 3 1 1 3 3 3 3 3 3 3 3 1 3 1]
# Accuracy : 0.7166666666666667
