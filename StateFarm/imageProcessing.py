from scipy import misc, io
import numpy as np
import matplotlib.pyplot as plot
import os

def rgbToGray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.597, 0.114])

images = []
i = 0

basePath = './Images/train'
for file in os.listdir(basePath):
    dirPath = basePath + '/' + file
    for imageFile in os.listdir(dirPath):
        image = misc.imread(dirPath + '/' + imageFile)
        gray = rgbToGray(image)
        gray = misc.imresize(gray, 0.4)
        gray = gray.ravel()
        images.append(gray)
        i += 1
        print (i)
    io.savemat(file + '.mat', {'images': images})


