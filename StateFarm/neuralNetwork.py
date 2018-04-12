import numpy as np
from scipy import io
import os
import re
import math

def loadAllData(folderName):
    X = np.empty((0,1024))
    Y = np.empty((0,1))
    Xcv = np.empty((0, 1024))
    Ycv = np.empty((0, 1))

    for file in os.listdir(folderName):
        filePath = folderName + '/' + file
        data = io.loadmat(filePath)
        currentData = np.array(data['z'])
        numOfRows = currentData.shape[0]
        trainRows = int(math.ceil(numOfRows * 0.8))

        X = np.concatenate((X, currentData[0:trainRows, :]), axis=0)
        Xcv = np.concatenate((Xcv, currentData[trainRows:numOfRows, :]), axis=0)

        regex = re.compile(r'\d+')
        yVal = regex.findall(file)[0]
        currentY = np.full((trainRows, 1), int(yVal), dtype=int)
        Y = np.concatenate((Y, currentY), axis=0)
        currentY = np.full((numOfRows-trainRows, 1), int(yVal), dtype=int)
        Ycv = np.concatenate((Ycv, currentY), axis=0)

    return(X, Xcv, Y, Ycv)



def randomInitialization(rows, cols, epsilon):
    randomArray = np.random.rand(rows, cols)
    return (randomArray * 2 * epsilon) - epsilon


