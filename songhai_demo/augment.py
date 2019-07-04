import pandas as pd

# df = pd.read_csv('sample_submission.csv')
# df_orign = df.copy()
# df[df[:] > -100] = 10
# df[df[:] <= -100] = 0
# df_new = df_orign.join(df.iloc[:, 2:], lsuffix='_caller', rsuffix='_other')
# df_new.to_csv('augmented_train_new_data.csv', sep=',', header=True, index=False)

df = pd.read_csv('sample_submission.csv')
# df[df[:] == -126.23] = -500.0
df_origin = df.copy()
df_new = df.iloc[:, 2:]


def mapper(label):
    row = df_new.iloc[label, :]
    idx = row.idxmax(axis=1)
    row[:] = 0
    row[:][idx] = 80
    return True


df_new.select(mapper)
df_new = df_origin.join(df_new, lsuffix='_caller', rsuffix='_other')
df_new.to_csv('augmented_test_new_data_argmax.csv', sep=',', header=True, index=False)
