import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import joblib as jlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# load training data set
trainFilePath = '../resource/train_new.csv'
trainDataSet = pd.read_csv(filepath_or_buffer=trainFilePath)
trainDataSet = trainDataSet.rename(columns=lambda x: x.strip())
data_num = len(trainDataSet)
print(data_num)  
print(trainDataSet.shape)  

# prepare train data set
# use ndarray type to save data
trainInputlist = np.array(trainDataSet.iloc[:, 2:])
trainInputlistWithX = np.array(
    trainDataSet.drop(['y'], axis=1, inplace=False))
trainXlist = np.array(trainDataSet.iloc[:, 0])
trainYlist = np.array(trainDataSet.iloc[:, 1])
print("train data set prepared")


if os.path.isfile("../output/modelX_Classifier.mdl"):
    print("model X file exists!")
    # load trained X model
    modelX = jlib.load("../output/modelX_Classifier.mdl")
    print("load model X finished!")
else:
    print("model X file not exist, create it and save!")
    # create XGBClassifier model
    modelX = xgb.XGBClassifier(max_depth=9,
                               learning_rate=0.1,
                               n_estimators=70,
                               silent=False,
                               objective='multi:softmax',
                               num_class=320)

    # train model X
    modelX.fit(trainInputlist, trainXlist)
    print("train model X finished!")
    jlib.dump(modelX, "../output/modelX_Classifier.mdl")


# load testing data set
testFilePath = '../resource/sample_submission.csv'
testDataSet = pd.read_csv(filepath_or_buffer=testFilePath)
testDataSet = testDataSet.rename(columns=lambda x: x.strip())
test_num = len(testDataSet)
print(test_num)  
print(testDataSet.shape)

# prepare test data set
# use ndarray type to save data
testInputlist = np.array(testDataSet.iloc[:, 2:])
testXlist = np.array(testDataSet.iloc[:, 0])
testYlist = np.array(testDataSet.iloc[:, 1])
Test_XList = testXlist.tolist()
Test_YList = testYlist.tolist()
print("test data set prepared")

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
modelY = xgb.XGBClassifier(max_depth=9,
                          learning_rate=0.1,
                          n_estimators=70,
                          silent=False,
                          objective='multi:softmax',
                          num_class=160)

# train model Y
modelY.fit(trainInputlistWithX, trainYlist)
print("train model Y finished!")

# predict the result
# add Pred_XList into testInputList to create testInputlistWithX
testInputlistWithX = np.c_[np.array(Pred_XList), testInputlist]
ansY = modelY.predict(testInputlistWithX)
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
