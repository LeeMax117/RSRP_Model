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

def val_in_test_set(epoch_num):
    # load test set and predict
    test_full = pd.read_csv("D:\Work\AI_competation\dataset\\sample_submission.csv")
    test = DataSet(test_full)

    while test.epochs_completed == 0:
        test_data, test_label = test.next_batch(100, is_training=False)
        inputs = Variable(torch.Tensor(test_data))
        outputs = net(inputs)
        ground_truth = Variable(torch.Tensor(test_label))

        try:
            predict = torch.cat((predict, outputs), 0)
            labels = torch.cat((labels, ground_truth), 0)
        except NameError:
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

    plt.scatter(x_label, y_label)
    plt.scatter(x_pre, y_pre)
    xlabel('x')
    ylabel('y')
    title('%d epochs'%(epoch_num))
    text(150, 80,'MSE_score:%.3f'%(MSE_score),fontsize=15)
    plt.show()

    return MSE_score

net = Net()
# define loss

criterion = torch.nn.MSELoss()
lr = 0.01
batch_size = 50
# create your optimizer
for epoch in range(30):  # loop over the dataset multiple times

    optimizer = optim.Adam(net.parameters(), lr=lr)
    current_epoch = train_data.epochs_completed
    steps = 0
    print('epoch completed:%d , learning_rate: %.3f'%(current_epoch,lr))

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
        total_loss = loss + (7e-2) * l2_loss
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
    batch_size *= 1.0
    lr = lr * 0.9
    test_score = val_in_test_set(current_epoch)
    try:
        if test_score < best_score:
            best_score = test_score
            torch.save(net.state_dict(), 'D:\Work\AI_competation\yuyang_Demo\ckpt\\best_model')
            print('**********model saved*************')
    except NameError:
        best_score = test_score
        torch.save(net.state_dict(), 'D:\Work\AI_competation\yuyang_Demo\ckpt\\best_model')

print('Finished Training')

# save the model
torch.save(net.state_dict(),'D:\Work\AI_competation\yuyang_Demo\ckpt\\final_model')