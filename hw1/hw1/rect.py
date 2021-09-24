# rect.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layer, hid):
        super(Network, self).__init__()
        assert(layer == 1 or layer == 2)
        self.layer = layer
        self.input = nn.Linear(2, hid)
        self.output = nn.Linear(hid, 1)
        if layer == 2:
            self.hid = nn.Linear(hid, hid)

    def forward(self, input):
        th = nn.Tanh()
        sg = nn.Sigmoid()
        if self.layer == 1:
            x = self.input(input)
            x = th(x)
            x = self.output(x)
            output = sg(x)
        else:
            x = self.input(input)
            x = th(x)
            x = self.hid(x)
            x = th(x)
            x = self.output(x)
            output = sg(x)
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    assert(layer == 1 or layer == 2)
    xrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-8,end=8.1,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)
    # INSERT CODE HERE
    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net.input(grid)
        th = nn.Tanh()
        output = th(output)
        if layer == 2:
            output = net.hid(output)
            output = th(output)
        # output = (output >= 0).float()
        net.train() # toggle batch norm, dropout back again

        pred = (output[:, node] >= 0).float()
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]),
                       cmap='Wistia', shading='auto')