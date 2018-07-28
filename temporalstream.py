import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import os
import numpy as np
from skimage import io, transform
import math
import time
import collections
from utils import *
from floss import floss
from data.STdatas import STTrainData, STValData
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-7, required=False)
parser.add_argument('--loss_save', default='loss_temporal.png', required=False)
parser.add_argument('--save_name', default='best_temporal.pth.tar', required=False)
parser.add_argument('--save_path', default='save', required=False)
parser.add_argument('--loss_function', default='f', required=False)
parser.add_argument('--num_epoch', type=int, default=10, required=False)
parser.add_argument('--device', default='0')
parser.add_argument('--resume', type=int, default=0, help='0 from vgg, 1 from pretrained model.')
parser.add_argument('--pretrained_model', default='save/best_spatial.pth.tar', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=16, required=False)
args = parser.parse_args()

device = torch.device('cuda:'+args.device)

STTrainLoader = DataLoader(dataset=STTrainData, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

STValLoader = DataLoader(dataset=STValData, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

class VGG(nn.Module):
    
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(256, 128, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(128, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 1, kernel_size=1, padding=0),
                                     )

        self.final = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        x = self.final(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def save_checkpoint(state,filename,save_path):
    torch.save(state, os.path.join(save_path, filename))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    for i, sample in enumerate(train_loader):
        input = sample['flow']
        target = sample['gt']
        input = input.float().to(device)
        target = target.to(device)
        output = model(input)
        target = target.view(output.size())
        loss = criterion(output, target)
        loss_mini_batch += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        losses.update(loss_mini_batch, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        loss_mini_batch = 0
        if (i+1) % 5000 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses))
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input = sample['flow']
            target = sample['gt']
            input = input.float().to(device)
            target = target.to(device)
            output = model(input)
            target = target.view(output.size())

            outim = output.cpu().data.numpy().squeeze()
            targetim = target.cpu().data.numpy().squeeze()
            aae1, auc1, _ = computeAAEAUC(outim, targetim)
            auc.update(auc1)
            aae.update(aae1)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 1000 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses,))
    print ('AUC: {0}\t AAE: {1}'.format(auc.avg, aae.avg))
    return losses.avg

# main

if args.resume == 1:
    print('building model and loading from pretrained model...')
    model = VGG(make_layers(cfg['D']))
    trained_model = args.pretrained_model
    pretrained_dict = torch.load(trained_model)
    epochold = pretrained_dict['epoch']
    pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    print('done!')
else:
    print('building model and loading weights from vgg...')
    model = VGG(make_layers(cfg['D']))
    in_channels = 20
    pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
    model_dict = model.state_dict()
    new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
    new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    print('done!')

if args.loss_function != 'f':
    criterion = torch.nn.BCELoss().to(device)
else:
    criterion = floss().to(device)
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Training and testing loop
train_loss = []
val_loss = []
for epoch in tqdm(range(args.num_epoch)):
    loss1 = train(STTrainLoader, model, criterion, optimizer, epoch)
    train_loss.append(loss1)
    loss1 = validate(STValLoader, model, criterion, epoch)
    val_loss.append(loss1)
    plot_loss(train_loss, val_loss, os.path.join(args.save_path, args.loss_save))
    print('epoch%05d, val loss is: %05f' % (epoch, loss1))
    save_checkpoint({'epoch': epoch, 'arch': 'flow', 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),},
                            '%05d'%epoch+args.save_name, args.save_path)
















