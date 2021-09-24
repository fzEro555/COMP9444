# kuzu.py


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.input = nn.Linear(28 * 28, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        m = nn.LogSoftmax(dim=-1)
        x = self.input(x)
        output = m(x)
        return output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.input = nn.Linear(28 * 28, 120)
        self.hidden = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        a1 = nn.Tanh()
        output = a1(self.input(x))
        a2 = nn.LogSoftmax(dim=-1)
        output = a2(self.hidden(output))
        return output # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 12, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(4, stride=1)
        self.lin1 = nn.Linear(8000, 10)

    def forward(self, x):
        r = nn.ReLU()
        x = r(self.conv1(x))
        x = self.maxpool1(x)
        x = r(self.conv2(x))
        x = self.maxpool2(x)
        #print(x.shape)
        x = x.view(-1, 8000)
        x = self.lin1(x)
        a2 = nn.LogSoftmax(dim=-1)
        output = a2(x)
        return output 