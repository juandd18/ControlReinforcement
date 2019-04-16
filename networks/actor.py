import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class FCBody(nn.Module):
    def __init__(self, state_dim, gate=F.relu):
        super(FCBody, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 88)
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))


    def forward(self, x):
        m = nn.ELU()
        x = m(self.bn1(self.fc1(x)))
        x = m(self.fc2(x))
        return x