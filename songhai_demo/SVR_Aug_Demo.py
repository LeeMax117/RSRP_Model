import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os

load_model = False
cv_search = False

# read train and test data.
df_train = pd.read_csv('augmented_train_new_data_argmax_80.csv')
df_test = pd.read_csv('augmented_test_new_data_argmax_80.csv')

# split data into signal strength and location data.
offline_location = df_train.iloc[:, :2]
offline_signal = df_train.iloc[:, 2:]
online_location = df_test.iloc[:, :2]
online_signal = df_test.iloc[:, 2:]

# location aggregation (optional)
off_location = offline_location.iloc[:, 0] + 0.001 * offline_location.iloc[:, 1]
on_location = online_location.iloc[:, 0] + 0.001 * online_location.iloc[:, 1]

# train / validation / optimization / evaluation
if os.path.exists('test_model.m') and load_model:
    print('start loading the model!!!')
    mod = joblib.load('test_model.m')
    # predicting
    preds = mod.predict(online_signal)
    mse = metrics.mean_squared_error(np.round(preds), online_location.iloc[:, 0])
    print('the mse is %f' % mse)
elif cv_search:
    print('start cv search!!!')
    para_c = {'C': range(65, 70)}
    svr_rbf = SVR(kernel='rbf', gamma='auto')
    reg = GridSearchCV(svr_rbf, para_c)
    reg.fit(offline_signal, off_location)
    scores = reg.cv_results_['mean_test_score']
    # plotting the scores
    plt.plot(range(1, scores.shape[0] + 1), scores, '-o', linewidth=2.0)
    plt.xlabel('C')
    plt.ylabel('score')
    plt.grid(True)
    plt.show()
else:
    print('start training model!!!')
    svr_rbf = SVR(kernel='rbf', C=70, gamma='auto').fit(offline_signal, off_location)
    # predicting
    preds = svr_rbf.predict(online_signal)
    mse = metrics.mean_squared_error(np.round(preds), online_location.iloc[:, 0])
    print('the mse is %f' % mse)
    # save model
    joblib.dump(reg, 'test_model.m')