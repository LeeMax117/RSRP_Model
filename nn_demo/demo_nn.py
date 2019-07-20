import torch
from torch.autograd import Variable
import torch.nn
import torch.optim as optim
from pylab import *
import pandas as pd # data processing, CSV file I/O
from utils import *

# input the dataset
# path to where the data lies
#dpath = './data/'
# full_data = pd.read_csv("D:\Work\AI_competation\dataset\\train.csv")
full_data = pd.read_csv("D:\Work\AI_competation\yuyang_Demo\\train_new.csv")
train_data = DataSet(full_data)

def val_in_test_set():
    # load test set and predict
    test_full = pd.read_csv("D:\Work\AI_competation\dataset\\sample_submission.csv")
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
    print('last outputs:(%.3f,%.3f), last ground truth:(%.3f,%.3f)' % \
          (np_outputs[-1][0], np_outputs[-1][1], test_label[-1][0], test_label[-1][1]))
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
ckpt_path = 'D:\Work\AI_competation\yuyang_Demo\ckpt\\best_model'
net.load_state_dict(torch.load(ckpt_path))

criterion = torch.nn.MSELoss()
lr = 3e-4
batch_size = 512
# create your optimizer
best_score, predict, test_label = val_in_test_set()
for epoch in range(10000):  # loop over the dataset multiple times

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
            np_outputs = outputs.detach().numpy()
            print('last outputs:(%.3f,%.3f), last ground truth:(%.3f,%.3f)' % \
                  (np_outputs[-1][0], np_outputs[-1][1], labels[-1][0], labels[-1][1]))
    print('##########epoch end#################')
    print('loss: %.3f,total_loss:%.3f' % (loss, total_loss))
    np_outputs = outputs.detach().numpy()
    print('last outputs:(%.3f,%.3f), last ground truth:(%.3f,%.3f)' % \
          (np_outputs[-1][0], np_outputs[-1][1], labels[-1][0], labels[-1][1]))
    print('####################################')
    # batch_size *= 1.0
    '''
    lr = lr * 0.99
    if current_epoch > 30:
        if lr > 3e-5:
            lr = lr * 0.999
        else:
            lr = 3e-5
    '''
    test_score, predict, test_label = val_in_test_set()
    if test_score < best_score:
        best_score = test_score
        torch.save(net.state_dict(), 'D:\Work\AI_competation\yuyang_Demo\ckpt\\best_model')
        print('**********model saved*************')
        draw_scatter(current_epoch, test_score, predict, test_label)

print('Finished Training')

# save the model
torch.save(net.state_dict(),'D:\Work\AI_competation\yuyang_Demo\ckpt\\final_model')