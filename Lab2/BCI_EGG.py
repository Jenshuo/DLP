
from dataloader import read_bci_data
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

# Termnal : python BCI_RGG.py "train or test"   --> argv[0] : BCI_EEG.py  argv[1]: train or test
mode = sys.argv[1]  # train or test

# hyperparameter
batch_size = 64
learning_rate = 1e-3
epoches = 300

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

class EEG_Net(nn.Module):
    def __init__(self, i):
        super(EEG_Net, self).__init__()
        self.i = i

        # Define activation function
        if self.i == 0:
            self.act = nn.ELU(alpha = 1.0)
        elif self.i == 1:
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

        # First conv
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1,1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        # deepwise conv
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act, 
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        # separable conv
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.act, 
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.75)
        )

        # Linear
        self.fc = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out


def train(epoch, act):
    eegnet.to(device)
    eegnet.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Mini batch gradient descent
    for step, (batch_x, batch_y) in enumerate(loader_train, 0):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        #  print('\n\n epoch: ', epoch, '| step: ', step, '| batch x: ', batch_x.numpy().shape, '| batch_y: ', batch_y.numpy().shape)

        # zero the parameter gradients
        Optimizer.zero_grad()

        # forward + backward + optimize
        y_train = eegnet(batch_x)
        loss = Loss(y_train, batch_y)
        loss.backward()
        Optimizer.step()

        # running_loss += loss.item()
        # calculate accuracy
        total += batch_y.size(0)
        correct += (y_train.max(1)[1] == batch_y).sum().item()

    # print accuracy
    acc_train = 100 * correct / total
    print(act +" Train %d accuracy = %d %%" %(epoch, acc_train))

    # Save model
    # print("Saving %d th model" %(epoch))
    torch.save(eegnet.state_dict(), PATH)

    return acc_train, eegnet.state_dict()


def test(epoch, act):
    eegnet.to(device)
    eegnet.eval()
    if mode == 'train':
        eegnet.load_state_dict(torch.load(PATH))
    else:
        eegnet.load_state_dict(torch.load(PATH_best))

    running_loss = 0.0
    correct = 0
    total = 0

    for step, (batch_x, batch_y) in enumerate(loader_test, 0):
        # print(batch_x.size())
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_test = eegnet(batch_x)
        loss = Loss(y_test, batch_y)
        # running_loss += loss.item()
        total += batch_y.size(0)
        correct += (y_test.max(1)[1] == batch_y).sum().item()

    # print accuracy
    acc_test = 100 * correct / total
    print(act+ " Test %d accuracy = %d %%" %(epoch, acc_test))

    return acc_test


if __name__ == "__main__":

    if mode == "train":

        for i in range(3):  # 3 kind of activation
            # Path
            if i == 0:
                PATH = './eeg_net_elu.pth'
                PATH_best = './egg_net_best_elu.pth'
                act = "elu"
                acc_elu_train = []
                acc_elu_test = []
            elif i == 1:
                PATH = './eeg_net_relu.pth'
                PATH_best = './egg_net_best_relu.pth'
                act = "relu"
                acc_relu_train = []
                acc_relu_test = []
            else:
                PATH = './eeg_net_leakyrelu.pth'
                PATH_best = './egg_net_best_leakyrelu.pth'
                act = "leakyrelu"
                acc_leakyrelu_train = []
                acc_leakyrelu_test = []

            eegnet = EEG_Net(i)
            Loss = nn.CrossEntropyLoss()    # Loss function
            Optimizer = optim.Adam(eegnet.parameters(), lr=learning_rate, weight_decay=1e-6)   # Optimizer
        
            best_acc = 0

            acc_train = []
            acc_test = []
            for epoch in range(epoches):
                train_acc, eegnet_state_dict = train(epoch, act)
                acc_train.append(train_acc)
                test_acc = test(epoch, act)
                acc_test.append(test_acc)
                # Find the best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(eegnet_state_dict, PATH_best)

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

        print("BEST ACC elu = %d %%" %(elu_best))
        print("BEST ACC relu = %d %%" %(relu_best))
        print("BEST ACC leakyrelu = %d %%" %(leakyrelu_best))

        # Plot 
        plt.title("Activation function comparision(EEGNet)")
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


    if mode == 'test':
        for i in range(3):
            # Path
            if i == 0:
                PATH_best = './egg_net_best_elu.pth'
                act = "elu"
            elif i == 1:
                PATH_best = './egg_net_best_relu.pth'
                act = "relu"
            else:
                PATH_best = './egg_net_best_leakyrelu.pth'
                act = "leakyrelu"

            eegnet = EEG_Net(i)
            Loss = nn.CrossEntropyLoss()    # Loss function
            Optimizer = optim.Adam(eegnet.parameters(), lr=learning_rate, weight_decay=1e-6)   # Optimizer

            test(0, act)
            
            
