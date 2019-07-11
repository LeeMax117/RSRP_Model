import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# load training data set
trainFilePath = '../resource/train_new.csv'
trainDataSet = pd.read_csv(filepath_or_buffer=trainFilePath)
trainDataSet = trainDataSet.rename(columns=lambda x: x.strip())
data_num = len(trainDataSet)
print(data_num) 
print(trainDataSet.shape) 

# combine x and y: use x as integral part, use y as fractional part
trainXlistWithY = []
# combine y and x: use y as integral part, use x as fractional part
trainYlistWithX = []

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
    trainXlistWithY.append(trainDataSet.iloc[row]['x'] +
                           0.001 * trainDataSet.iloc[row]['y'])
    trainYlistWithX.append(trainDataSet.iloc[row]['y'] +
                           0.001 * trainDataSet.iloc[row]['x'])

# load testing data set
testFilePath = '../resource/sample_submission.csv'
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

# create XGBRegressor model
modelX = xgb.XGBRegressor(max_depth=11,
                         learning_rate=0.1,
                         n_estimators=400,
                         silent=False,
                         objective='reg:squarederror')

# train model X
modelX.fit(Train_InputList, trainXlistWithY)
print("train model X finished!")

# predict the result
ansX = modelX.predict(Test_InputList)
print(ansX)

# transfer and round result
Pred_XList = []
for index in range(0, test_num):
    Pred_XList.append(int(ansX[index]))  # 这里不应该用round
print(Pred_XList)
accuracy = accuracy_score(Test_XList, Pred_XList)
print("X accuarcy: %.2f%%" % (accuracy*100.0))

# create XGBRegressor model
modelY = xgb.XGBRegressor(max_depth=11,
                         learning_rate=0.1,
                         n_estimators=400,
                         silent=False,
                         objective='reg:squarederror')

# train model Y
modelY.fit(Train_InputList, trainYlistWithX)
print("train model Y finished!")

# predict the result
ansY = modelY.predict(Test_InputList)
print(ansY)

# transfer and round result
Pred_YList = []
for index in range(0, test_num):
    Pred_YList.append(int(ansY[index]))  
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
