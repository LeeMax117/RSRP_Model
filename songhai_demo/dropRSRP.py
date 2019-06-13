import pandas as pd

df = pd.read_csv('train.csv')


def mapper(label):
    row = df.iloc[label, 2:]
    for item in row:
        if item > -126.0:
            return True
    return False


new_df = df.select(mapper)
print(df.describe())
print(new_df.describe())
new_df.to_csv('train_new.csv', sep=',', header=True, index=False)
