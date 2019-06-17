import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import metrics


# prepare data
df = pd.read_csv('train_new_2.csv')
df2 = pd.read_csv('sample_submission.csv')
offline_location, offline_rsrp = np.array(df.iloc[:, [0, 1]]), np.array(df.iloc[:, 2:])
online_location, online_rsrp = np.array(df2.iloc[:, [0, 1]]), np.array(df2.iloc[:, 2:])


# accuracy metrics
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions-labels)**2, 1)))


# # knn regression
# knn_reg = neighbors.KNeighborsRegressor(8, weights='uniform', metric='euclidean')
# predictions = knn_reg.fit(offline_rsrp, offline_location).predict(online_rsrp)
# acc = accuracy(predictions, online_location)
# print(np.c_[online_location, predictions])
# print("accuracy: %f m" % acc)

# # knn classification
# labels = np.round(offline_location[:, 0] / 100.0) * 100 + np.round(offline_location[:, 1] / 100.0)
# knn_cls = neighbors.KNeighborsClassifier(8, weights='uniform', metric='euclidean')
# predict_labels = knn_cls.fit(offline_rsrp, labels).predict(online_rsrp)
# x = np.floor(predict_labels/100.0)
# y = predict_labels - x * 100
# predictions = np.column_stack((x, y)) * 100
# acc = accuracy(predictions, online_location)
# print("accuracy: %f m" % acc)

from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
parameters = {'n_neighbors': range(1,50)}
knn_reg = neighbors.KNeighborsRegressor(weights='uniform', metric='euclidean')
clf = GridSearchCV(knn_reg, parameters)
clf.fit(offline_rsrp, offline_location)
scores = clf.cv_results_['mean_test_score']
k = np.argmax(scores)
print("the optimal args: %d" % k)

# plot the curve
import matplotlib.pyplot as plt
plt.plot(range(1, scores.shape[0] + 1), scores, '-o', linewidth=2.0)
plt.xlabel('k')
plt.ylabel('score')
plt.grid(True)
plt.show()


# knn regression
knn_reg = neighbors.KNeighborsRegressor(k+1, weights='uniform', metric='euclidean')
predictions = knn_reg.fit(offline_rsrp, offline_location).predict(online_rsrp)
acc = accuracy(predictions, online_location)
np.savetxt('pred.txt',predictions, fmt="%d")
print(np.c_[online_location, predictions])
mse = metrics.mean_squared_error(online_location, predictions)
print(mse)
