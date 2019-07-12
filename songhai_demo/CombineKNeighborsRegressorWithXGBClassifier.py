import numpy as np
import pandas as pd
import xgboost as xgb
from math import sqrt
from collections import Counter
from sklearn import neighbors
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# define a drop mapper.
class DropMapper(object):
    curdf = None

    def __init__(self, df):
        self.curdf = df

    def mapper(self, label):
        if self.curdf is not None:
            cnt = 0
            row = self.curdf.iloc[label, 2:]
            for item in row:
                if item > -125.0:
                    cnt += 1
                if cnt > 6:
                    return False
            return True
        else:
            return True


# prepare data
df_train_knn = pd.read_csv('../resource/train_new_1.csv')
df_test = pd.read_csv('../resource/sample_submission.csv')
total_test_cnt = len(df_test)

# data pre-precessing
drop_test = DropMapper(df_test)
df_test_knn = df_test.select(drop_test.mapper)
df_test_xgb = df_test.append(df_test_knn)
df_test_xgb.drop_duplicates(keep=False, inplace=True)
print(df_test_knn.shape)
print(df_test_xgb.shape)
offline_location, offline_rsrp = np.array(df_train_knn.iloc[:, [0, 1]]), np.array(df_train_knn.iloc[:, 2:])
online_location, online_rsrp = np.array(df_test_knn.iloc[:, [0, 1]]), np.array(df_test_knn.iloc[:, 2:])

# train model
knn_reg = neighbors.KNeighborsRegressor(4, weights='uniform', metric='euclidean')
predictions = knn_reg.fit(offline_rsrp, offline_location).predict(online_rsrp)
knn_preds = predictions.astype(int)
mse = metrics.mean_squared_error(online_location, predictions)
mse2 = metrics.mean_squared_error(online_location, knn_preds)
# print(mse, mse2)
print("KNN XY MSE: %.2f" % mse)
print("KNN XY MSE after integration: %.2f" % mse2)

# plot the result
plt.figure()
plt.scatter(knn_preds[:, 0], knn_preds[:, 1])
plt.scatter(online_location[:, 0], online_location[:, 1], alpha=0.5)
plt.ylabel('KNN real and predict')
plt.show()

###################################
# begin to handle strong signal points
trainFilePath = '../resource/train_new.csv'
trainDataSet = pd.read_csv(filepath_or_buffer=trainFilePath)
trainDataSet = trainDataSet.rename(columns=lambda x: x.strip())
data_num = len(trainDataSet)
print(data_num)
print(trainDataSet.shape)

# prepare train data set
# use ndarray type to save data
trainInputlist = np.array(trainDataSet.iloc[:, 2:])
trainXlist = np.array(trainDataSet.iloc[:, 0])
trainYlist = np.array(trainDataSet.iloc[:, 1])
print("train data set prepared")

# check testing data set
test_num = len(df_test_xgb)
print(test_num)  # 205
print(df_test_xgb.shape)

# prepare test data set
# use ndarray type to save data
testInputlist = np.array(df_test_xgb.iloc[:, 2:])
testXlist = np.array(df_test_xgb.iloc[:, 0])
testYlist = np.array(df_test_xgb.iloc[:, 1])
Test_XList = testXlist.tolist()
Test_YList = testYlist.tolist()
print("test data set prepared")

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

# predict the result
ansX = modelX.predict(testInputlist)
print(ansX)

# transfer and round result
Pred_XList = []
for index in range(0, test_num):
    Pred_XList.append(ansX[index])
print(Pred_XList)
accuracy = accuracy_score(Test_XList, Pred_XList)
print("XGB X accuarcy: %.2f%%" % (accuracy*100.0))

# create XGBClassifier model
modelY = xgb.XGBClassifier(max_depth=9,
                          learning_rate=0.1,
                          n_estimators=70,
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
    Pred_YList.append(ansY[index])
print(Pred_YList)
accuracy = accuracy_score(Test_YList, Pred_YList)
print("XGB Y accuarcy: %.2f%%" % (accuracy*100.0))

# draw real and predict points
plt.scatter(Pred_XList, Pred_YList, linewidths=0)
plt.scatter(Test_XList, Test_YList, linewidths=0)
plt.ylabel('XGB real and predict')
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
print("XGB XY MSE: %.2f" % MSE_XY)


# calculate total MSE
KNN_Point_Cnt = len(online_location)
print("KNN Point Cnt: %d" % KNN_Point_Cnt)
for index in range(0, KNN_Point_Cnt):
    tmp_list = []
    tmp_list.append(knn_preds[index][0])
    tmp_list.append(knn_preds[index][1])
    Pred_XList.append(knn_preds[index][0])
    Pred_YList.append(knn_preds[index][1])
    Pred_XYList.append(tmp_list)

for index in range(0, KNN_Point_Cnt):
    tmp_list = []
    tmp_list.append(online_location[index][0])
    tmp_list.append(online_location[index][1])
    Test_XList.append(online_location[index][0])
    Test_YList.append(online_location[index][1])
    Real_XYList.append(tmp_list)

print("Pred_XYList:")
print(Pred_XYList)
print("Real_XYList:")
print(Real_XYList)
Total_MSE_XY = mean_squared_error(Real_XYList, Pred_XYList)
print("Total XY MSE: %.2f" % Total_MSE_XY)

# draw total real and predict points
plt.scatter(Pred_XList, Pred_YList, linewidths=0)
plt.scatter(Test_XList, Test_YList, linewidths=0)
plt.ylabel('Total real and predict')
plt.show()

# calculate distance
SSE_XYList = []
Distance_XYList = []
for index in range(0, total_test_cnt):
    SSE_XY = (Test_XList[index] - Pred_XList[index]) ** 2 +\
             (Test_YList[index] - Pred_YList[index]) ** 2
    SSE_XYList.append(SSE_XY)
    Distance_XY = sqrt(SSE_XY)
    Distance_XYList.append(Distance_XY)
print("Distance_XYList is:")
print(Distance_XYList)

# round float to int
Distance_IntList = []
for index in range(0, total_test_cnt):
    Distance_IntList.append(round(Distance_XYList[index]))
print("Distance_IntList is:")
print(Distance_IntList)

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

# draw PDF histogram
plt.hist(Distance_IntList, bins=Max_Distance, range=None, density=True,
         cumulative=False, histtype='bar')
plt.show()

# draw CDF histogram
plt.hist(Distance_IntList, bins=Max_Distance, range=None, density=True,
         cumulative=True, histtype='bar')
plt.show()



