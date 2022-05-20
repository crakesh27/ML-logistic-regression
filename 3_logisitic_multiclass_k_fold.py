import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import math


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
        if(loss >= prev_loss+10e-4):
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
stride = 0
for i in range(5):  # K folds
    train1 = dataset[0:stride]
    train2 = dataset[stride+30:150]
    train = train1.append(train2, ignore_index=True)
    test = dataset[stride:stride+30]
    stride += 30

    X_train = train.iloc[:, 0:4].values
    X_train = np.insert(X_train, 0, np.ones(120), 1)
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
    X_test = np.insert(X_test, 0, np.ones(30), 1)
    Y_test = test.iloc[:, 4].values

    # Logistic Regression one vs all
    print("Logistic Regression one vs all , fold-", end='')
    print(i+1)

    np.random.seed(100)
    theta1 = np.random.rand(5)
    theta1 = logistic_regression(X_train, Y_train_1, theta1, 0.00000001, 100)
    print("Weights 1 vs rest: ", end='')
    print(theta1)

    np.random.seed(100)
    theta2 = np.random.rand(5)
    theta2 = logistic_regression(X_train, Y_train_2, theta2, 0.00000001, 100)
    print("Weights 2 vs rest: ", end='')
    print(theta2)

    np.random.seed(100)
    theta3 = np.random.rand(5)
    theta3 = logistic_regression(X_train, Y_train_3, theta3, 0.00000001, 100)
    print("Weights 3 vs rest: ", end='')
    print(theta3)

    Y_pred = predict_data(X_test, Y_test, theta1, theta2, theta3)

    TP, FP, TN, FN = perf_measure(Y_test, Y_pred)
    print("Overall Accuracy : ", end='')
    print((TP+TN)/(TP+TN+FP+FN))

# Output
# Logistic Regression one vs all , fold-1
# Weights 1 vs rest: [ 0.5336352   0.2209477   0.39467101  0.80768366 -0.00724908]
# Weights 2 vs rest: [ 0.53373887  0.22155703  0.39498773  0.80807727 -0.00712209]
# Weights 3 vs rest: [ 0.53410807  0.22372703  0.39611565  0.80947902 -0.00666981]
# Overall Accuracy : 0.76666666666666666
# Logistic Regression one vs all , fold-2
# Weights 1 vs rest: [ 0.53446311  0.22676012  0.39705127  0.8126228  -0.00530345]
# Weights 2 vs rest: [ 0.53339213  0.22057878  0.39376157  0.80877173 -0.00650384]
# Weights 3 vs rest: [ 0.53364105  0.22201545  0.39452617  0.8096668  -0.00622484]
# Overall Accuracy : 0.7
# Logistic Regression one vs all , fold-3
# Weights 1 vs rest: [ 0.53363507  0.22124822  0.39467878  0.80773205 -0.00715153]
# Weights 2 vs rest: [ 0.53385874  0.22255594  0.39536191  0.80858013 -0.00687978]
# Weights 3 vs rest: [ 0.53398891  0.223317    0.39575947  0.80907369 -0.00672162]
# Overall Accuracy : 0.65
# Logistic Regression one vs all , fold-4
# Weights 1 vs rest: [ 0.53327708  0.21907919  0.39398208  0.80585982 -0.00770466]
# Weights 2 vs rest: [ 0.53433576  0.22527686  0.397174    0.80992779 -0.00640601]
# Weights 3 vs rest: [ 0.53386871  0.22254271  0.39576586  0.80813317 -0.00697892]
# Overall Accuracy : 0.7333333333333333
# Logistic Regression one vs all , fold-5
# Weights 1 vs rest: [ 0.53375614  0.22176306  0.39485556  0.8084323  -0.00691599]
# Weights 2 vs rest: [ 0.5339772   0.22305995  0.39553513  0.80926496 -0.00664943]
# Weights 3 vs rest: [ 0.53374818  0.2217164   0.3948311   0.80840234 -0.00692559]
# Overall Accuracy : 0.76666666666666664
