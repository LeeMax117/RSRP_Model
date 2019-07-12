import torch
from torch.autograd import Variable
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd # data processing, CSV file I/O
from dataprocess import DataSet

# input the dataset
# path to where the data lies
#dpath = './data/'
full_data = pd.read_csv("D:\Work\AI_competation\dataset\\train_new.csv")
train_data = DataSet(full_data)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        '''
        self.conv1 = torch.nn.Conv2d(2, 20, (1, 1))
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=1)
        self.BN1 = torch.nn.BatchNorm2d(20, 1, 6)
        self.conv2 = torch.nn.Conv2d(20, 200, (1, 6))
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=1)
        self.BN2 = torch.nn.BatchNorm2d(200, 1, 1)
        self.fc1 = torch.nn.Linear(200, 30)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=1)
        self.fc2 = torch.nn.Linear(30, 2)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=1)
        '''
        # for the first 1*1 and 1*6 conv
        self.conv1 = torch.nn.Conv2d(2, 20, (1, 1))
        self.BN1 = torch.nn.BatchNorm2d(20, 1, 6)
        self.conv2 = torch.nn.Conv2d(20, 200, (1, 6))

        # the bypass of 1*6 and 1*1 conv
        self.conv_1 = torch.nn.Conv2d(2,20,(1,6))
        self.BN_1 = torch.nn.BatchNorm2d(20,1,2)
        self.conv_2 = torch.nn.Conv2d(20, 200, (1,1))

        # softmax layer to identify the position possibility
        self.softmax = torch.nn.Linear(400, 20)

        # full connected
        self.BN2 = torch.nn.BatchNorm2d(400, 1, 1)
        self.fc1 = torch.nn.Linear(400, 100)
        self.fc2 = torch.nn.Linear(100, 2)

        torch.nn.init.normal_(self.conv1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=1)

    def forward(self, x):
        # leaky_r = torch.nn.LeakyReLU(6e-10)
        # use 1*1 to specify which frequency and which position
        x_1 = self.conv1(x)
        x_1_reuse = x_1
        x_1 = torch.relu(self.BN1(x_1))
        # use 1*6*200 to specify 3 position (x,y) and 3 signal and 1 posibility, 20 choice make 200
        # x = F.relu(self.conv2(x))
        x_1 = self.conv2(x_1)

        x_2 = self.conv_1(x)
        x_2_reuse = x_2
        x_2 = torch.relu((self.BN_1(x_2)))
        x_2 = self.conv_2(x_2)
        x = torch.cat((x_1,x_2),1)
        x = torch.relu(self.BN2(x))
        x = x.view(-1, self.num_flat_features(x))
        p = torch.softmax(self.softmax(x),dim=1)
        print('shape of self.p',p.shape)
        print('example of p:',p[0])
        print('shape of x_reues:', x_1_reuse.shape)
        print('exmaple of x_reuse',x_1_reuse[0])
        x_1 = x_1_reuse * p
        x_2 = x_2_reuse * p
        x_1 = torch.relu(self.BN1(x_1))
        # use 1*6*200 to specify 3 position (x,y) and 3 signal and 1 posibility, 20 choice make 200
        x_1 = self.conv2(x_1)
        x_2 = torch.relu((self.BN_1(x_2)))
        x_2 = self.conv_2(x_2)
        x = torch.cat((x_1, x_2), 1)
        x = torch.relu(self.BN2(x))
        x = x.view(-1, self.num_flat_features(x))

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

net = Net()
# define loss

criterion = torch.nn.MSELoss()
lr = 0.01
batch_size = 50
# create your optimizer
for epoch in range(30):  # loop over the dataset multiple times

    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr = lr * 0.9

    current_epoch = train_data.epochs_completed
    steps = 0
    print('epoch completed:%d'%(current_epoch))

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
        # total_loss = loss + (3e-5) * l2_loss
        total_loss = loss
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
print('Finished Training')