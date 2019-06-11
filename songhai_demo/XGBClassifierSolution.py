import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# load training data set
trainFilePath = 'C:/laptop/PycharmProjects/MachineLearning/resource/train.csv'
trainDataSet = pd.read_csv(filepath_or_buffer=trainFilePath)
trainDataSet = trainDataSet.rename(columns=lambda x: x.strip())
data_num = len(trainDataSet)
print(data_num)  # 25811
print(trainDataSet.shape)  # (25811, 14)

# prepare feature data set
Train_InputList = []
Train_XList = []
Train_YList = []
for row in range(0, data_num):
    tmp_list = []
    tmp_list.append(trainDataSet.iloc[row]['2.1G(10)'])
    tmp_list.append(trainDataSet.iloc[row]['2.1G(11)'])
    tmp_list.append(trainDataSet.iloc[row]['2.1G(12)'])
    tmp_list.append(trainDataSet.iloc[row]['2.1G(4)'])
    tmp_list.append(trainDataSet.iloc[row]['2.1G(7)'])
    tmp_list.append(trainDataSet.iloc[row]['2.1G(8)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(10)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(11)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(12)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(4)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(7)'])
    tmp_list.append(trainDataSet.iloc[row]['3.5G(8)'])
    Train_InputList.append(tmp_list)
    Train_XList.append(trainDataSet.iloc[row]['x'])
    Train_YList.append(trainDataSet.iloc[row]['y'])

# transfer list to ndarray
trainInputlist = np.array(Train_InputList)
trainXlist = np.array(Train_XList)
trainYlist = np.array(Train_YList)
print("transfer list to ndarray")

# load testing data set
testFilePath = 'C:/laptop/PycharmProjects/MachineLearning/resource/sample_submission.csv'
testDataSet = pd.read_csv(filepath_or_buffer=testFilePath)
testDataSet = testDataSet.rename(columns=lambda x: x.strip())
test_num = len(testDataSet)
print(test_num)  # 205
print(testDataSet.shape)  # (205, 14)

# prepare feature data set
Test_InputList = []
Test_XList = []
Test_YList = []
for row in range(0, test_num):
    tmp_list = []
    tmp_list.append(testDataSet.iloc[row]['2.1G(10)'])
    tmp_list.append(testDataSet.iloc[row]['2.1G(11)'])
    tmp_list.append(testDataSet.iloc[row]['2.1G(12)'])
    tmp_list.append(testDataSet.iloc[row]['2.1G(4)'])
    tmp_list.append(testDataSet.iloc[row]['2.1G(7)'])
    tmp_list.append(testDataSet.iloc[row]['2.1G(8)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(10)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(11)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(12)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(4)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(7)'])
    tmp_list.append(testDataSet.iloc[row]['3.5G(8)'])
    Test_InputList.append(tmp_list)
    Test_XList.append(testDataSet.iloc[row]['x'])
    Test_YList.append(testDataSet.iloc[row]['y'])

# transfer list to ndarray
testInputlist = np.array(Test_InputList)
testXlist = np.array(Test_XList)
testYlist = np.array(Test_YList)
print("transfer list to ndarray")

# create XGBClassifier model
modelX = xgb.XGBClassifier(max_depth=8,
                          learning_rate=0.1,
                          n_estimators=40,
                          silent=False,
                          objective='multi:softmax',
                          num_class=320)

# train model X
modelX.fit(trainInputlist, trainXlist)
print("train model X finished!")

# predict the result
ansX = modelX.predict(testInputlist)
print(ansX)

# transfer and round result
Pred_XList = []
for index in range(0, test_num):
    Pred_XList.append(round(ansX[index]))
print(Pred_XList)
accuracy = accuracy_score(Test_XList, Pred_XList)
print("X accuarcy: %.2f%%" % (accuracy*100.0))

# create XGBClassifier model
modelY = xgb.XGBClassifier(max_depth=8,
                          learning_rate=0.1,
                          n_estimators=40,
                          silent=False,
                          objective='multi:softmax',
                          num_class=160)

# train model Y
modelY.fit(trainInputlist, trainYlist)
print("train model Y finished!")

# predict the result
ansY = modelY.predict(testInputlist)
print(ansY)

# transfer and round result
Pred_YList = []
for index in range(0, test_num):
    Pred_YList.append(round(ansY[index]))
print(Pred_YList)
accuracy = accuracy_score(Test_YList, Pred_YList)
print("Y accuarcy: %.2f%%" % (accuracy*100.0))

# draw real and predict points
plt.scatter(Pred_XList, Pred_YList, linewidths=0)
plt.scatter(Test_XList, Test_YList, linewidths=0)
plt.ylabel('real and predict')
plt.show()

# create predict XY list
Pred_XYList = []
for index in range(0, test_num):
    tmp_list = []
    tmp_list.append(Pred_XList[index])
    tmp_list.append(Pred_YList[index])
    Pred_XYList.append(tmp_list)

# create real XY list
Real_XYList = []
for index in range(0, test_num):
    tmp_list = []
    tmp_list.append(Test_XList[index])
    tmp_list.append(Test_YList[index])
    Real_XYList.append(tmp_list)

# calculate MSE
MSE_XY = mean_squared_error(Real_XYList, Pred_XYList)
print("XY MSE: %.2f" % MSE_XY)
