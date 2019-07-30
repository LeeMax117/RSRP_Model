import torch
import pandas as pd
from utils import *
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

net = Net()
base_dir = os.path.expanduser('D:\Work\AI_competation\submission')
test_path = os.path.join(base_dir,'sample_submission.csv')
ckpt_path = ckpt_dir = os.path.join(base_dir,'model_ckpt')
net.load_state_dict(torch.load(ckpt_path))

# load test set and predict
test_full = pd.read_csv(test_path)
test = DataSet(test_full)

while test.epochs_completed == 0:
    test_data, test_label = test.next_batch(test.num_examples, is_training=False)
    inputs = Variable(torch.Tensor(test_data))
    outputs = net(inputs)
    ground_truth = Variable(torch.Tensor(test_label))

predict = outputs.detach().numpy()
labels = ground_truth.detach().numpy()
distance_list = []
for index, i in enumerate(predict):
    distance_list.append(np.linalg.norm(predict[index] - labels[index]))

distance_list = np.array(distance_list)

x = []
CDF = []
max_dist = int(max(distance_list) + 1)
for num in range(10 * int(max_dist + 1)):
    x.append(num/10)
    CDF.append(sum(distance_list < num/10)/(len(distance_list)))

CDF_list = []
for i in range(int(max_dist+1)):
    CDF_list.append((i,sum(distance_list < i)))
print(CDF_list)

plt.bar(x, CDF,align="center", width=0.1, alpha=0.5)
plt.xticks([i/2 for i in range(max_dist*2 +2 )])
plt.show()
