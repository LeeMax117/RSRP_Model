import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn

class DataSet:
    def __init__(self,
                 full_data,
                 dtype=np.float64,
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
        full_data = (full_data + 126.23) / 72
        # full_data[full_data[:] == -1] = -1.2
        train_data = np.array(full_data, dtype=dtype)
        self._data = train_data.reshape(-1, 1, 2, 6)

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
        if self._epochs_completed == 0 and start == 0 and is_training:
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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # for the first 1*1 and 1*6 conv
        self.conv1 = torch.nn.Conv2d(1, 20, (2, 1))
        self.conv2 = torch.nn.Conv2d(20, 200, (1, 6))

        # the bypass of 1*6 and 1*1 conv
        self.conv_1 = torch.nn.Conv2d(1,20,(1,6))
        self.conv_2 = torch.nn.Conv2d(20, 200, (2,1))

        # full connected
        self.BN = torch.nn.BatchNorm2d(400, 1, 1)
        self.fc1 = torch.nn.Linear(400, 100)
        self.fc2 = torch.nn.Linear(100, 2)

        torch.nn.init.normal_(self.conv1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=1)
        # torch.nn.init.normal_(self.conv_2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=1)

    def forward(self, x):
        # leaky_r = torch.nn.LeakyReLU(6e-10)
        # use 1*1 to specify which frequency and which position
        x_1 = torch.relu(self.conv1(x))
        # x_1 = torch.relu(self.BN1(x_1))
        # use 1*6*200 to specify 3 position (x,y) and 3 signal and 1 posibility, 20 choice make 200
        # x = F.relu(self.conv2(x))
        x_1 = torch.tanh(self.conv2(x_1))
        x_2 = torch.relu(self.conv_1(x))
        # x_2 = torch.relu(self.BN_1(x_2))
        x_2 = torch.tanh(self.conv_2(x_2))
        x = torch.cat((x_1,x_2),1)
        # x = torch.relu(self.BN(x))
        x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    # dpath = './data/'
    full_data = pd.read_csv("D:\Work\AI_competation\dataset\\train.csv")
    train_data = DataSet(full_data)
    print(train_data.next_batch(1))