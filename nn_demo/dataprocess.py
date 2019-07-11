import numpy as np
import pandas as pd

class DataSet:
    def __init__(self,
                 full_data,
                 dtype=np.float32,
                 inference = False,
                 seed=None):
        np.random.seed(seed)
        if dtype not in (np.uint16, np.float32,np.float64):
            raise TypeError('Invalid image dtype %r, expected uint16 or float32/64' %
                            dtype)

        self._num_examples = full_data.shape[0]

        if not inference:
            # define the label
            label_x = np.array(full_data['x'], dtype=np.int16)
            label_y = np.array(full_data[' y'], dtype=np.int16)
            label = np.vstack((label_x, label_y))
            self._labels = label.T
        else:
            self._labels = None

        # transform the dataset from 1*12 to 2*6
        del full_data['x']
        del full_data[' y']
        # full_data = (full_data + 60.25)/(32.99) - 1
        # full_data.fillna()
        full_data = (full_data + 126.23) / 36 - 1
        full_data[full_data[:] == -1] = -100
        train_data = np.array(full_data, dtype=dtype)
        self._data = train_data.reshape(-1, 2, 1, 6)

        self._inference = inference

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inference(self):
        return self._inference

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def labels(self):
        if self._inference:
            return None
        else:
            return self._labels

    @property
    def data(self):
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def random_list(self):
        return self._perm

    def next_batch(self,
                   batch_size,
                   shuffle=True,
                   is_training=True):
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            perm0 =  np.random.permutation(self._num_examples)
            # shuffle the dataset
            if shuffle:
                self._data = self._data[perm0]
                self._labels = self._labels[perm0]
            self._perm = perm0

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            data_rest_part, labels_rest_part = self._data[start:], self._labels[start:]

            # Shuffle the data
            if shuffle:
                perm = np.random.permutation(self._num_examples)
                self._data = self._data[perm]
                self._labels = self._labels[perm]
                self._perm = perm

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            data_new_part, labels_new_part = self._data[start:end], self._labels[start:end]

            return np.concatenate((data_rest_part, data_new_part), axis=0), \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            return self._data[start:end], self._labels[start:end]

if __name__ == '__main__':
    # dpath = './data/'
    full_data = pd.read_csv("D:\Work\AI_competation\dataset\\train.csv")
    train_data = DataSet(full_data)
    print(train_data.next_batch(1))