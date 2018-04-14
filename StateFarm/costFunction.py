import numpy as np
import math
from sigmoid import sigmoid, sigmoidDerivative
from scipy import io

def costFunction(parameters, inputLayerSize, hiddenLayerSize, outputLayerSize, X, y, lambdaVal):
    theta1Total = hiddenLayerSize * (inputLayerSize + 1)
    theta2Total = outputLayerSize * (hiddenLayerSize + 1)

    theta1 = np.reshape(parameters[0:theta1Total], (hiddenLayerSize, (inputLayerSize + 1)), order="F")
    theta2 = np.reshape(parameters[theta1Total:], (outputLayerSize, (hiddenLayerSize + 1)), order="F")

    m = X.shape[0]

    reshapeY = np.zeros((m, outputLayerSize), dtype=int)
    for i in range(0,m):
        reshapeY[i, y[i,0]-1] = 1

    X = np.concatenate((np.ones((m,1), dtype=int), X), axis=1)
    z2 = X.dot(np.transpose(theta1))
    a2 = sigmoid(z2)

    a2 = np.concatenate((np.ones((m, 1), dtype=int), a2), axis=1)
    z3 = a2.dot(np.transpose(theta2))

    h = sigmoid(z3)

    sumOverNodes = np.sum((reshapeY * np.log(h)) + ((1 - reshapeY) * np.log(1 - h)),axis=1)
    cost = -1 * np.sum(sumOverNodes, axis=0) / m

    regularization = (sum(sum(theta1[:,1:] * theta1[:,1:])) + sum(sum(theta2[:,1:] * theta2[:,1:]))) * lambdaVal / (2 * m)
    cost = cost + regularization

    ##########################
    #####Back Propogation#####
    ##########################
    theta1Grad = np.zeros(theta1.shape, dtype=int)
    theta2Grad = np.zeros(theta2.shape, dtype=int)

    deltaThree = h - y
    deltaTwo = deltaThree.dot(theta2) * sigmoidDerivative(np.concatenate((np.ones((m, 1), dtype=int), z2), axis=1))

    theta1Grad = theta1Grad + np.transpose(deltaTwo[:,1:]).dot(X)
    theta1Grad = theta1Grad / m
    theta1Grad[:,1:] = theta1Grad[:,1:] + [(lambdaVal / m) * theta1[:,1:]]

    theta2Grad = theta2Grad + np.transpose(deltaThree).dot(a2)
    theta2Grad = theta2Grad / m
    theta2Grad[:,1:] = theta2Grad[:,1:] + [(lambdaVal / m) * theta2[:,1:]]
    print(cost)

def CostFunctionTest():
    data = io.loadmat('TestData/ex4data1.mat')
    X = data['X']
    y = data['y']

    data = io.loadmat('TestData/ex4weights.mat')
    theta1 = data['Theta1']
    theta2 = data['Theta2']
    theta1 = (theta1.T).ravel()
    theta2 = (theta2.T).ravel()
    params = np.concatenate((theta1, theta2), axis=0)

    costFunction(params, 400, 25, 10, X, y, 1)

CostFunctionTest()