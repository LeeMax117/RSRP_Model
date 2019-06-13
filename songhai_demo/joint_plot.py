import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file IO
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# cmap = getattr(cm, 'plasma_r', cm.hot_r)
df = pd.read_csv('train.csv')
X = np.array(df)[:, 0].astype(np.int)  # get x position and y position
Y = np.array(df)[:, 1].astype(np.int)
Z_full = np.array(df)[:, 2:]  # get features
v_min = Z_full.min()
v_max = Z_full.max()
x_max = int(X.max())
y_max = int(Y.max())

columns = [col for col in df if len(col) > 2]  # get singal strength columns
col_size = len(columns)
row = math.floor(math.sqrt(col_size))  # calculate the plot array size
col = math.ceil(math.sqrt(col_size))


def f(x_pos, y_pos, v):
    # print(x_pos)
    mat = np.full((x_max+1, y_max+1), -126)  # create 2D array which will be used for contour plot
    mat[x_pos, y_pos] = v
    return mat


fig = plt.figure()
for ind in range(1, col_size+1):
    Z = f(X, Y, Z_full[:, ind-1])
    X_line = np.linspace(0, 5, 318)
    Y_line = np.linspace(0, 5, 157)
    X_axis, Y_axis = np.meshgrid(X_line, Y_line)
    plt.subplot(row, col, ind)
    # print(X.shape, Y.shape, Z.shape, Z.transpose().shape)
    C = plt.contour(X_axis, Y_axis, Z.transpose(), 20)
    # plt.clabel(C, inline=True, fontsize=12) # set the contour label
plt.show()
