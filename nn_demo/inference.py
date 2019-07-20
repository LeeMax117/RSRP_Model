import torch
import pandas as pd
from utils import *
from torch.autograd import Variable
from pylab import *

net = Net()
ckpt_path = 'D:\Work\AI_competation\yuyang_Demo\ckpt\\best_model(0.43)'
net.load_state_dict(torch.load(ckpt_path))
criterion = torch.nn.MSELoss()

# load test set and predict
test_full = pd.read_csv("D:\Work\AI_competation\dataset\\sample_submission.csv")
# test_full = pd.read_csv("D:\Work\AI_competation\yuyang_Demo\\test_new.csv")
test = DataSet(test_full)

while test.epochs_completed == 0:
    test_data, test_label = test.next_batch(test.num_examples, is_training=False)
    inputs = Variable(torch.Tensor(test_data))
    outputs = net(inputs)
    ground_truth = Variable(torch.Tensor(test_label))

    '''
    try:
        predict = torch.cat((predict, outputs), 0)
        labels = torch.cat((labels, ground_truth), 0)
    except NameError:
        predict = outputs
        labels = ground_truth
    '''
predict = outputs
labels = ground_truth
MSE_score = criterion(predict, labels)
np_outputs = outputs.detach().numpy()
print('last outputs:(%.3f,%.3f), last ground truth:(%.3f,%.3f)' % \
      (np_outputs[-1][0], np_outputs[-1][1], test_label[-1][0], test_label[-1][1]))
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
