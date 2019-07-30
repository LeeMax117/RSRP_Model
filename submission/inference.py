import torch
import pandas as pd
from utils import *
from torch.autograd import Variable
from pylab import *
import os

net = Net()
base_dir = os.path.expanduser('D:\Work\AI_competation\submission')
# change test_path for your test set to infer
test_path = os.path.join(base_dir,'sample_submission.csv')
ckpt_path = ckpt_dir = os.path.join(base_dir,'model_ckpt')
net.load_state_dict(torch.load(ckpt_path))
criterion = torch.nn.MSELoss()

# load test set and predict
test_full = pd.read_csv(test_path)
test = DataSet(test_full)

while test.epochs_completed == 0:
    test_data, test_label = test.next_batch(test.num_examples, is_training=False)
    inputs = Variable(torch.Tensor(test_data))
    outputs = net(inputs)
    ground_truth = Variable(torch.Tensor(test_label))

predict = outputs
labels = ground_truth
MSE_score = criterion(predict, labels)
print('MSE_score:', MSE_score)

predict = predict.detach().numpy()
test_label = labels.detach().numpy()
# plot the result
x_label = test_label[:, 0]
y_label = test_label[:, 1]

x_pre = predict[:, 0]
y_pre = predict[:, 1]

plt.scatter(x_label, y_label,linewidths=0.1)
plt.scatter(x_pre, y_pre,linewidths=0.1)
xlabel('x')
ylabel('y')
title('test_set')
text(150, 80,'MSE_score:%.3f'%(MSE_score),fontsize=15)
plt.show()
