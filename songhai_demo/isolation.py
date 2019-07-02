import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

graph_dict = {}


def union(val1, val2):
    root1 = getroot(val1)
    root2 = getroot(val2)
    if root1 == root2:
        return True
    else:
        graph_dict[root2] = root1
        return False


def getroot(val):
    if val not in graph_dict:
        graph_dict[val] = val
        return val
    root = val
    while root != graph_dict[root]:
        root = graph_dict[root]
    return root


npdata = np.loadtxt('train.csv', skiprows=1, dtype=int, usecols=(0, 1), delimiter=',')
train_set = set(zip(npdata[:, 0], npdata[:, 1]))
max_x, max_y = np.max(npdata[:, 0]), np.max(npdata[:, 1])
allset = set([(i, j) for i in range(max_x+1) for j in range(max_y+1)])
emptyset = allset.difference(train_set)
for j in range(max_y+2):
    for i in range(max_x+1):
        if (i, j) not in train_set:
            if (i, j-1) not in train_set:
                union((i, j-1), (i, j))
            if (i-1, j) not in train_set:
                union((i-1, j), (i, j))
isolation = []
root = getroot((0, max_y+1))
for cor in emptyset:
    if root == getroot(cor):
        isolation.append(cor)

if len(isolation) > 0:
    X, Y = zip(*isolation)
    plt.figure()
    plt.xlim(-2, 320)
    plt.ylim(-2, 159)
    plt.scatter(X, Y)
    plt.show()
    new_data = np.c_[np.array(X), np.array(Y)]
    np.savetxt('isolation.csv', new_data, fmt='%d', delimiter=',')