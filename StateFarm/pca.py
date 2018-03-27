from scipy import misc, io, linalg
import numpy as np
import matplotlib.pyplot as plot
import os
from PIL import Image

def fsvd(X, k, i):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    if (X.shape[0] < X.shape[1]):
        X = np.transpose(X)
        isTransposed = True
    else:
        isTransposed = False

    n = X.shape[1]
    l = k + 2
    G = np.random.randn(n, l)


    prev = X.dot(G)
    H = np.array(prev)
    for j in range(1,i+1):
        i = X.dot(X.T.dot(prev))
        H = np.hstack([H, i])
        prev = i

    Q = np.linalg.qr(H, 'economic')

    T = np.transpose(X).dot(Q)

    [Vt, St, W] = np.linalg.svd(T, full_matrices=False)

    Ut = Q.dot(W)

    if isTransposed:
        V = Ut[:,0:k]
        U = Vt[:,0:k]
    else:
        U = Ut[:,0:k]
        V = Vt[:,0:k]
    S = np.diag(St)
    S = S[0:k,0:k]
    return (U, S, V)


def pca(X):

    (m,n) = X.shape
    newX = (X.T.dot(X))/m
    [U,S,V] = np.linalg.svd(newX, full_matrices=False)
data = io.loadmat('./Data/c0.mat')
images = data['images']
images = np.array(images)

[U, S, V] = fsvd(images[0:500,:], 10, 2)

img1 = (U.dot(S)).dot(V.T)

plot.imshow(img1)
plot.show()
print(images.shape[0])