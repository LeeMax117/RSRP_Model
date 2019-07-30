import torch
from torch.autograd import Variable
import torch.nn
import torch.optim as optim
from pylab import *
import pandas as pd # data processing, CSV file I/O
from utils import *
import os

##############################################
############# train parameters ###############
##############################################
load_ckpt = False
train_epoch_num = 5
lr = 0.01
batch_size = 64
base_dir = os.path.expanduser('D:\Work\AI_competation\submission')
train_path = os.path.join(base_dir,'train.csv')
ckpt_dir = os.path.join(base_dir,'train_ckpt')
ckpt_path = os.path.join(ckpt_dir,'saved_model')
test_path = os.path.join(base_dir,'sample_submission.csv')

# input the dataset
full_data = pd.read_csv(train_path)
train_data = DataSet(full_data)

def val_in_test_set():
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
    np_outputs = outputs.detach().numpy()
    print('MSE_score:', MSE_score)

    # trans the predict and test_label to numpy
    predict = predict.detach().numpy()
    test_label = labels.detach().numpy()

    return MSE_score,predict,test_label

def draw_scatter(epoch_num, MSE_score,predict,test_label):
    # plot the result
    x_label = test_label[:, 0]
    y_label = test_label[:, 1]

    x_pre = predict[:, 0]
    y_pre = predict[:, 1]

    plt.scatter(x_label, y_label,linewidths=0.5)
    plt.scatter(x_pre, y_pre,linewidths=0.5)
    xlabel('x')
    ylabel('y')
    title('%d epochs'%(epoch_num))
    text(150, 80,'MSE_score:%.3f'%(MSE_score),fontsize=15)
    plt.show()

net = Net()
# define loss

# load the current best model
if load_ckpt:
    net.load_state_dict(torch.load(ckpt_path))

criterion = torch.nn.MSELoss()
# create your optimizer
best_score, predict, test_label = val_in_test_set()
for epoch in range(train_epoch_num):  # loop over the dataset multiple times

    optimizer = optim.Adam(net.parameters(), lr=lr)
    current_epoch = train_data.epochs_completed
    steps = 0
    print('epoch completed:%d , learning_rate: %.5f'%(current_epoch,lr))

    while current_epoch == train_data.epochs_completed:
        # wrap them in Variable
        data, labels = train_data.next_batch(int(batch_size))
        inputs = Variable(torch.Tensor(data))
        outputs = net(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        ground_truth = Variable(torch.Tensor(labels))
        loss = criterion(outputs, ground_truth)
        l1_loss = 0
        l2_loss = 0
        for param in net.parameters():
            l1_loss += torch.sum(abs(param))
            l2_loss += torch.sum(param ** 2)
        total_loss = loss + (3e-4) * l2_loss
        # total_loss = loss
        total_loss.backward()
        optimizer.step()
        steps += 1

        # print statistics
        if steps % 100 == 0:    # print every 500 mini-batches
            print('loss: %.3f,total_loss:%.3f' % (loss, total_loss))
    print('##########epoch end#################')
    print('loss: %.3f,total_loss:%.3f' % (loss, total_loss))
    print('####################################')

    # batch_size *= 1.0
    if lr > 3e-4:
        lr = lr * 0.99
    else:
        lr = 3e-4

    test_score, predict, test_label = val_in_test_set()
    if test_score < best_score:
        best_score = test_score
        torch.save(net.state_dict(), ckpt_path)
        print('**********model saved*************')
        draw_scatter(current_epoch, test_score, predict, test_label)

print('Finished Training')

# save the model
final_model_path = os.path.join(ckpt_dir,'final_model')
torch.save(net.state_dict(),final_model_path)