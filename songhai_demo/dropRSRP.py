import pandas as pd

df = pd.read_csv('train.csv')
threshold = 3


def mapper(label):
    cnt = 0
    row = df.iloc[label, 2:]
    for item in row:
        if item > -126.0:
            cnt += 1
        if item > -100.0 or cnt >= threshold:
            return True
    return False


new_df = df.select(mapper)
print(df.describe())
print(new_df.describe())
new_df.to_csv('train_new_2.csv', sep=',', header=True, index=False)
