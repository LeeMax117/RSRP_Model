import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file IO

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('train.csv')
data.head()
data.info()
print(data.isnull().sum())
print(data.describe())

# histgram
fig = plt.figure()
plt.subplot(121)
sns.distplot(data.x.values, bins=30, kde=True)
plt.xlabel('x values', fontsize=12)
plt.title('x spatial distribution')
plt.subplot(122)
sns.distplot(data[' y'].values, bins=30, kde=True)
plt.xlabel('y values', fontsize=12)
plt.title('y spatial distribution')

# scatter map
plt.figure()
plt.scatter(range(data.shape[0]), data[' 2.1G(11)'], color='purple')

# count plot
plt.figure()
sns.countplot(data.x.values % 10)
plt.xlabel('index of each signal strength')

# get the name of all the columns
cols = data.columns
print(cols)

# calculate the pearson co-efficient for all combinations
data_corr = data.corr()

# heat map
plt.figure()
sns.heatmap(data_corr, annot=True)
# mask unimportant features
sns.heatmap(data_corr, mask=data_corr < 1, cbar=False)
plt.show()

# set the threshold to select only highly correlated attributes
threshold = 0.5
# list of pairs along with correlation above threshold
corr_lst = []
size = data_corr.shape[0]

for i in range(size):
    for j in range(i + 1, size):
        if threshold <= data_corr.iloc[i, j] < 1 or data_corr.iloc[i, j] <= -threshold:
            corr_lst.append((data_corr.iloc[i, j], i, j))

# sort to show higher ones first
s_corr_lst = sorted(corr_lst, key=lambda x: -abs(x[0]))

for v, i, j in s_corr_lst:
    print('%s and %s = %.2f' % (cols[i], cols[j], v))

# scatter plot of only the highly correlated pairs
# for v, i, j in s_corr_lst:
#    plt.figure()
#    sns.pairplot(data, size=6, x_vars=cols[i], y_vars=cols[j])

# data preparation
y = data[' y'].values
X = data.drop(['x', ' y'], axis=1)

cols = X.columns

# data split
from sklearn.model_selection import train_test_split

# random sample 20% as test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2)

# data pre processing
from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

# linear regression
from sklearn.linear_model import LinearRegression
print(np.isnan(X_train).any(), np.isnan(y_train).any())
lr = LinearRegression()
lr.fit(X_train, y_train)

# prediction
y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

# check the weight on each column
fs = pd.DataFrame({'columns': list(cols), 'coef': list(lr.coef_.T)})
fs.sort_values(by=['coef'], ascending=False)
print(fs)

# model metrics
from sklearn.metrics import r2_score
# test set
print('The r2 score of Linearregression on test is', r2_score(y_test, y_test_pred))
# train set
print('The r2 score of LinearRegression on train is', r2_score(y_train, y_train_pred))

f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(y_train-y_train_pred, bins=40, label='Residuals linear', color='y', alpha=0.5)
ax.set_title('Histogram of Residuals', fontsize=12)
ax.legend(loc='best')

fig_res = plt.figure()
sns.distplot(y_train - y_train_pred, bins=30, kde=True)
plt.xlabel('Histogram of Residuals', fontsize=12)
plt.show()
