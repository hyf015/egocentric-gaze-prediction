import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


batch_size = 1

class lstmnet(nn.Module):
    def __init__(self, num_channel=512, num_layer=2):
        super(lstmnet, self).__init__()
        self.lstm = nn.LSTM(num_channel, num_channel, num_layer)
        self.tanh = nn.Tanh()
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.lin = nn.Linear(512,512)
        self.relu = nn.ReLU()


    def forward(self, input, hidden):
        # this hidden should be (h, c)
        input = self.tanh(input)
        if hidden is None:
            ihidden = Variable(torch.zeros(self.num_layer, batch_size, self.num_channel)).to(input.device)
            icell = Variable(torch.zeros(self.num_layer, batch_size, self.num_channel)).to(input.device)
            inithidden = (ihidden, icell)
            out, hidden = self.lstm(input, inithidden)
        else:
            out, hidden = self.lstm(input, hidden)
        out = self.lin(out)
        return (self.relu(out), hidden)
