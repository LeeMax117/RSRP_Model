import numpy as np
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt

# prepare predicted x,y value
pred_data = np.loadtxt('../output/pred_Classifier.txt', dtype=int, skiprows=0, comments='#')
Pred_XList = pred_data[:, 0]
Pred_YList = pred_data[:, 1]
x_cnt = len(Pred_XList)
y_cnt = len(Pred_YList)
print(x_cnt)  # 205
print(y_cnt)  # 205
Pred_XYList = []
for index in range(0, x_cnt):
    tmp_list = []
    tmp_list.append(Pred_XList[index])
    tmp_list.append(Pred_YList[index])
    Pred_XYList.append(tmp_list)
print("Pred_XYList is:")
print(Pred_XYList)

# read true x,y value
import pandas as pd
testFilePath = 'C:/laptop/PycharmProjects/MachineLearning/resource/sample_submission.csv'
testDataSet = pd.read_csv(filepath_or_buffer=testFilePath)
testDataSet = testDataSet.rename(columns=lambda x: x.strip())
test_num = len(testDataSet)
print(test_num)  # 205行

Real_XYList = []
Real_XList = []
Real_YList = []
for row in range(0, test_num):
    tmp_list = []
    tmp_list.append(testDataSet.iloc[row][0])
    tmp_list.append(testDataSet.iloc[row][1])
    Real_XYList.append(tmp_list)
    Real_XList.append(testDataSet.iloc[row][0])
    Real_YList.append(testDataSet.iloc[row][1])
print("Real_XYList is:")
print(Real_XYList)

# calculate distance
SSE_XYList = []
Distance_XYList = []
for index in range(0, x_cnt):
    SSE_XY = (Real_XList[index] - Pred_XList[index]) ** 2 +\
             (Real_YList[index] - Pred_YList[index]) ** 2
    SSE_XYList.append(SSE_XY)
    Distance_XY = sqrt(SSE_XY)
    Distance_XYList.append(Distance_XY)
print("Distance_XYList is:")
print(Distance_XYList)
"""
Distance_XYList is:
[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.1622776601683795, 8.54400374531753, 7.615773105863909, 2.23606797749979, 7.0710678118654755, 1.4142135623730951, 2.23606797749979, 3.0, 2.23606797749979, 2.0, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 1.0, 1.0, 1.0, 2.23606797749979, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.4142135623730951, 2.8284271247461903, 3.1622776601683795, 1.0, 1.4142135623730951, 1.0, 1.0, 1.4142135623730951, 2.23606797749979, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 2.23606797749979, 1.4142135623730951, 1.0, 1.4142135623730951, 5.0, 1.0, 3.1622776601683795, 2.23606797749979, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 2.23606797749979, 2.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.4142135623730951, 1.0, 0.0, 1.4142135623730951, 1.4142135623730951, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.23606797749979, 1.0, 1.4142135623730951, 1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 2.23606797749979, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 2.23606797749979, 3.1622776601683795, 1.4142135623730951, 1.0, 1.4142135623730951, 2.23606797749979, 1.0, 2.23606797749979, 2.23606797749979, 1.4142135623730951, 1.0, 1.4142135623730951, 2.23606797749979, 3.1622776601683795]
"""

# round float to int
Distance_IntList = []
for index in range(0, test_num):
    Distance_IntList.append(round(Distance_XYList[index]))
print("Distance_IntList is:")
print(Distance_IntList)
"""
Distance_IntList is:
[2, 1, 1, 1, 1, 1, 3, 9, 8, 2, 7, 1, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 5, 1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 1, 2, 4, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 3, 2, 1, 1, 3, 0, 2, 1, 2, 3, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 3]
"""

# count element
Distance_Count = Counter(Distance_IntList)
print("Distance_Count is:")
print(Distance_Count)
Distance_Count = sorted(Distance_Count.items())
print("Distance_Count is:")
print(Distance_Count)
Max_Distance = max(Distance_IntList)
print("Max_Distance is:")
print(Max_Distance)
"""
Distance_Count is:
Counter({1: 158, 2: 26, 3: 11, 0: 5, 9: 1, 8: 1, 7: 1, 5: 1, 4: 1})
Distance_Count is:
[(0, 5), (1, 158), (2, 26), (3, 11), (4, 1), (5, 1), (7, 1), (8, 1), (9, 1)]
Max_Distance is:
9
"""

# draw PDF histogram
plt.hist(Distance_IntList, bins=Max_Distance, range=None, density=True,
         cumulative=False, histtype='bar')
plt.show()

# draw CDF histogram
plt.hist(Distance_IntList, bins=Max_Distance, range=None, density=True,
         cumulative=True, histtype='bar')
plt.show()


