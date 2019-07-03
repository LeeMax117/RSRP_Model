import pandas as pd


df = pd.read_csv('train.csv')
df_orign = df.copy()
df[df[:] > -100] = 1
df[df[:] <= -100] = 0
df_new = df_orign.join(df.iloc[:, 2:], lsuffix='_caller', rsuffix='_other')
df_new.to_csv('augmented_train_data.csv', sep=',', header=True, index=False)
