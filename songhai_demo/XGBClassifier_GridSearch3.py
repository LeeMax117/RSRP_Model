import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend

if __name__ == "__main__":
    # load training data set
    trainFilePath = '../resource/train.csv'
    # trainFilePath = '../resource/train_new.csv'
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

    # define parameters
    parameters = {
        'max_depth': [5, 6, 7, 8, 9, 10],     
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'n_estimators': [30, 40, 50, 60, 70, 80]
    }

    # create XGBClassifier model
    modelX = xgb.XGBClassifier(max_depth=9,
                              learning_rate=0.1,
                              n_estimators=70,
                              silent=False,
                              objective='multi:softmax',
                              num_class=320)

    # search best value
    gsearch = GridSearchCV(modelX,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',  # scoring='accuracy'
                           n_jobs=-1,
                           cv=3,
                           verbose=3)
    with parallel_backend('multiprocessing'):
        gsearch.fit(trainInputlist, trainXlist)

    # print best score and parameter value
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # draw parameter and score curve
    scores = gsearch.cv_results_['mean_test_score']
    plt.plot(range(1, scores.shape[0] + 1), scores, '-o', linewidth=2.0)
    plt.xlabel('candidate')
    plt.ylabel('score')
    plt.grid(True)
    plt.show()
    print(scores.shape[0])
