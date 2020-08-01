
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from dataloader import read_bci_data


# Terminal : python Deep_Conv.py train/test
mode = sys.argv[1]  # train or test

# hyperparameter
batch_size = 64
learning_rate = 1e-2
epoches = 150

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# Load data
train_data, train_label, test_data, test_label = read_bci_data()
loader_train = DataLoader(TensorDataset(torch.tensor(train_data.astype(np.float32)), torch.tensor(train_label.astype(np.long))), batch_size=batch_size)
loader_test = DataLoader(TensorDataset(torch.tensor(test_data.astype(np.float32)), torch.tensor(test_label.astype(np.long))), batch_size=batch_size)

# print(type(loader_test))
# dataiter = iter(loader_test)
# images, labels = dataiter.next()
# print(dataiter.next())
# print(type(images), type(labels))
# print(images.size(), labels.size())

class Deep_Conv_Net(nn.Module):
    def  __init__(self, i):
        super(Deep_Conv_Net, self).__init__()
        self.i = i

        # Define activation function
        if self.i == 0:
            self.act = nn.ELU(alpha=1.0)
        elif self.i == 1:
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

        # First conv
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2,1), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.75)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.75)
        )
        self.fc = nn.Linear(in_features=8600, out_features=2, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def train(epoch, act):
    deep_net.to(device)
    deep_net.train()

    correct = 0
    total = 0

    for step, (batch_x, batch_y) in enumerate(loader_train, 0):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        Optimizer.zero_grad()

        y_train = deep_net(batch_x)
        loss = Loss(y_train, batch_y)
        loss.backward()
        Optimizer.step()

        total += batch_y.size(0)
        correct += (y_train.max(1)[1] == batch_y).sum().item()
    
    acc_train = 100 * correct / total
    print(act + " Train %d accuracy = %d %%" %(epoch, acc_train))

    torch.save(deep_net.state_dict(), PATH)

    return acc_train, deep_net.state_dict()

def test(epoch, act):
    deep_net.to(device)
    deep_net.eval()

    if mode == "train":
        deep_net.load_state_dict(torch.load(PATH))
    else:
        deep_net.load_state_dict(torch.load(PATH_best))
    
    correct = 0
    total = 0

    for step, (batch_x, batch_y) in enumerate(loader_test, 0):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_test = deep_net(batch_x)
        loss = Loss(y_test, batch_y)
        total += batch_y.size(0)
        correct += (y_test.max(1)[1] == batch_y).sum().item()

    acc_test = 100 * correct / total
    print(act + " Test %d accuracy = %d %%" %(epoch, acc_test))

    return acc_test

if __name__ == "__main__":

    if mode == "train":

        for i in range(3):
            # Path
            if i == 0:
                PATH = './deep_elu.pth'
                PATH_best = './deep_best_elu.pth'
                act = "elu"
                acc_elu_train = []
                acc_elu_test = []
            elif i == 1:
                PATH = './deep_relu.pth'
                PATH_best = './deep_best_relu.pth'
                act = "relu"
                acc_relu_train = []
                acc_relu_test = []
            else:
                PATH = './deep_leakyrelu.pth'
                PATH_best = './deep_best_leakyrelu.pth'
                act = "leakyrelu"
                acc_leakyrelu_train = []
                acc_leakyrelu_test = []

            deep_net = Deep_Conv_Net(i)     # initialize net
            Loss = nn.CrossEntropyLoss()    # Loss function
            Optimizer = optim.Adam(deep_net.parameters(), lr=learning_rate)
            # Scheduler = optim.lr_scheduler.StepLR(Optimizer, step_size=30, gamma=0.05)

            best_acc = 0
            acc_train = []
            acc_test = []

            for epoch in range(epoches):
                train_acc, deep_state_dict = train(epoch, act)
                acc_train.append(train_acc)
                test_acc = test(epoch, act)
                acc_test.append(test_acc)

                # Find the best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(deep_state_dict, PATH_best)
                
                # Scheduler.step()
            
            if i == 0:
                acc_elu_train = acc_train
                acc_elu_test = acc_test
                elu_best = best_acc
            elif i == 1:
                acc_relu_train = acc_train
                acc_relu_test = acc_test
                relu_best = best_acc
            else:
                acc_leakyrelu_train = acc_train
                acc_leakyrelu_test = acc_test
                leakyrelu_best = best_acc
        

        print("Train ACC elu = %d %%" %(max(acc_elu_train)))
        print("Train ACC relu = %d %%" %(max(acc_relu_train)))
        print("Train ACC leakyrelu = %d %%" %(max(acc_leakyrelu_train)))

        print("BEST ACC elu = %d %%" %(elu_best))
        print("BEST ACC relu = %d %%" %(relu_best))
        print("BEST ACC leakyrelu = %d %%" %(leakyrelu_best))

         # Plot 
        plt.title("Activation function comparision(DeepNet)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(acc_elu_train, label='ELU train')
        plt.plot(acc_elu_test, label='ELU test')
        plt.plot(acc_relu_train, label='RELU train')
        plt.plot(acc_relu_test, label='RELU test')
        plt.plot(acc_leakyrelu_train, label='LEAKYRELU train')
        plt.plot(acc_leakyrelu_test, label='LEADYRELU test')
        plt.legend(loc='lower right')
        plt.show()

    if mode == "test":
        
        for i in range(3):
            if i == 0:
                PATH_best = './deep_best_elu.pth'
                act = "elu"
            elif i == 1:
                PATH_best = './deep_best_relu.pth'
                act = "relu"
            else:
                PATH_best = './deep_best_leakyrelu.pth'
                act = "leakyrelu"
            
            deep_net = Deep_Conv_Net(i)
            Loss = nn.CrossEntropyLoss()
            Optimizer = optim.Adam(deep_net.parameters(), lr=learning_rate, weight_decay=1e-6) 

            test(0, act)
