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


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    elif h is None:
        return None
    else:
        return tuple(repackage_hidden(v) for v in h)

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
            ihidden = Variable(torch.zeros(self.num_layer, batch_size, self.num_channel)).cuda(async = True)
            icell = Variable(torch.zeros(self.num_layer, batch_size, self.num_channel)).cuda(async = True)
            inithidden = (ihidden, icell)
            out, hidden = self.lstm(input, inithidden)
        else:
            out, hidden = self.lstm(input, hidden)
        out = self.lin(out)
        return (self.relu(out), hidden)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch, net, loader, criterion, optimizer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    net.train()
    hidden = None
    tanh = nn.Tanh()
    for i, sample in enumerate(loader):
        #print i
        inp = sample[0]
        #see = inp[0,0,:].numpy()
        #plt.plot(see)
        #plt.show()
        gt = sample[1]
        #(batch,10,512)
        inp = inp.permute(1,0,2).contiguous()
        gt = gt.permute(1,0,2).contiguous()
        inp = Variable(inp).cuda(async = True)
        gt = Variable(gt).cuda(async = True)
        
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        out, hidden = net(inp, hidden)
        loss = criterion(out, tanh(gt))
        loss.backward()
        losses.update(loss.data[0], batch_size)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1)%500 == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(loader)+1, batch_time = batch_time, loss= losses))

if __name__ == '__main__': 
    #wTrainLoader = DataLoader(dataset=wTrainData, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    #wValLoader = DataLoader(dataset=wValData, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    net = lstmnet()
    net.cuda()
    inp = Variable(torch.randn(16,1,512)).cuda()
    hidden = None
    o,h = net(inp, hidden)
    print o.size()
    '''
    criterion = torch.nn.MSELoss().cuda()
    lr = 1e-3
    for epoch in tqdm(range(1000)):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train(epoch, net, wTrainLoader, criterion, optimizer)
        if epoch%100 == 0:
            torch.save(net.state_dict(), 'savelstm/3layer/'+str(epoch)+'lstm.pth.tar')
            lr = lr/10
    '''
