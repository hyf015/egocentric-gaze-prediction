import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo

import os
import numpy as np
from skimage import io
from scipy import ndimage
import math
import time
import collections
from tqdm import tqdm

from floss import floss
from data.STdatas import STTrainData, STValData
from utils import *

##############################################################spatialtemporal data loader#######################################################

STTrainLoader = DataLoader(dataset=STTrainData, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

STValLoader = DataLoader(dataset=STValData, batch_size=10, shuffle=False, num_workers=1, pin_memory=True)

##############################################################spatialtemporal data loader#######################################################


class VGG_st_3dfuse(nn.Module):
    def __init__(self, features_s, features_t):
        super(VGG_st_3dfuse, self).__init__()
        self.features_t = features_t
        self.features_s = features_s
        self.relu = nn.ReLU()
        self.fusion = nn.Conv3d(512, 512, kernel_size=(1,3,3), padding=(0,1,1))
        self.pool3d = nn.MaxPool3d(kernel_size=(2,1,1), padding=0)
        self.bn = nn.BatchNorm3d(512)
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding = 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding = 1),
                                        nn.ReLU(inplace=True),
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

    def forward(self, x_s, x_t):
        x_s = self.features_s(x_s)
        x_t = self.features_t(x_t)
        x_s = x_s.unsqueeze(2)
        x_t = x_t.unsqueeze(2)
        x_fused = torch.cat((x_s, x_t), 2)
        x_fused = self.fusion(x_fused)
        
        x_fused = self.pool3d(x_fused)
        x_fused = x_fused.squeeze_(2)
        x_fused = self.bn(x_fused)

        x_fused = self.relu(x_fused)
        x_fused = self.decoder(x_fused)
        x = self.final(x_fused)
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

def train(st_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    for i, sample in enumerate(st_loader):
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        input_s = input_s.float().cuda(async=True)
        input_t = input_t.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var_s = Variable(input_s)
        input_var_t = Variable(input_t)
        target_var = Variable(target)
        output = model(input_var_s, input_var_t)
        target_var = target_var.view(output.size())
        loss = criterion(output, target_var)
        loss_mini_batch += loss.data[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        losses.update(loss.data[0], input_s.size(0))
        end = time.time()
        loss_mini_batch = 0
        if (i+1)%1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(st_loader)+1, batch_time = batch_time, loss= losses))

def validate(st_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    model.eval()
    end = time.time()
    for i, sample in enumerate(st_loader):
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        input_s = input_s.float().cuda(async=True)
        input_t = input_t.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var_s = Variable(input_s)
        input_var_t = Variable(input_t)
        target_var = Variable(target)
        output = model(input_var_s, input_var_t)
        target_var = target_var.view(output.size())
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input_s.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        outim = output.cpu().data.numpy().squeeze()
        targetim = target_var.cpu().data.numpy().squeeze()
        auc1 = computeAUC(outim,targetim)
        aae1 = computeAAE(outim,targetim)
        auc.update(auc1)
        aae.update(aae1)
        if i == 836:
            outim = output.data.cpu().numpy()
            outim = outim[0,:,:,:].squeeze()
            io.imsave(str(epoch)+'_fusion_test.jpg',outim)
            if not os.path.exists('targetfusion.jpg'):
                targetim = target_var.data.cpu().numpy()
                targetim = targetim[0,:,:,:].squeeze()
                io.imsave('targetfusion.jpg', targetim)
                print ('image saved!')
        if i % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(st_loader), batch_time=batch_time, loss=losses,))
    print ('AUC: {0}\t AAE: {1}'.format(auc.avg, aae.avg))
    return losses.avg


def computeAAE(output, target):
    res = []
    for batch in range(output.shape[0]):
        out_sq = output[batch,:,:].squeeze()
        tar_sq = target[batch,:,:].squeeze()
        predicted = ndimage.measurements.center_of_mass(out_sq)
        (i,j) = np.unravel_index(tar_sq.argmax(), tar_sq.shape)
        d = 112/math.tan(math.pi/6)
        r1 = np.array([predicted[0], predicted[1], d])
        r2 = np.array([i, j, d])
        angle = math.atan2(np.linalg.norm(np.cross(r1,r2)), np.dot(r1,r2))
        res.append(math.degrees(angle))
    if np.isnan(np.mean(res)):
        print 'aae:'
        print res
        return 0
    return np.mean(res)

def computeAUC(output, target):
    res = []
    for batch in range(output.shape[0]):
        out_sq = output[batch,:,:].squeeze()
        tar_sq = target[batch,:,:].squeeze()
        (i,j) = np.unravel_index(tar_sq.argmax(), tar_sq.shape)
        atgt = out_sq[i][j]
        fpbool = out_sq > atgt
        res.append(1 - float(fpbool.sum())/(out_sq.shape[0]*out_sq.shape[1]))
    if np.mean(res)<0:
        print 'auc:'
        print res
    return np.mean(res)

'''
if __name__ == '__main__':
    lr = 1e-7
    save_path = 'savefusion'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('building model...')
    in_channels = 20
    resume = 0
    # 2: resume from fusion
    # 0: from vgg
    # 1: resume from separately pretrained models
    if resume == 2:
        model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
        trained_model = 'savefusion/00004_fusion3d_bn_floss_checkpoint.pth.tar'
        pretrained_dict = torch.load(trained_model)
        epochnow = pretrained_dict['epoch']
        pretrained_dict = pretrained_dict['state_dict']
        model.cuda()
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif resume == 0:
        epochnow = 0
        model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
        #pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
        pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')

        model_dict_s = model.features_s.state_dict()
        model_dict_t = model.features_t.state_dict()
        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if 'features' in k}
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if 'features' in k}
        #print pretrained_dict.keys()
        for k in pretrained_dict.keys():
            pretrained_dict[k[9:]] = pretrained_dict[k]
        for k in new_pretrained_dict.keys():
            new_pretrained_dict[k[9:]] = new_pretrained_dict[k]
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict_t}
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict_s}

        model_dict_s.update(pretrained_dict)
        model_dict_t.update(new_pretrained_dict)
        model.features_s.load_state_dict(model_dict_s)
        model.features_t.load_state_dict(model_dict_t)
        model.cuda()
    else:
        epochnow = 0
        model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
        pretrained_dict_s = torch.load('save/00003_floss_checkpoint.pth.tar')
        pretrained_dict_t = torch.load('saveflow/00003__floss_checkpoint.pth.tar')
        model_dict_s = model.features_s.state_dict()
        model_dict_t = model.features_t.state_dict()
        pretrained_dict_s = pretrained_dict_s['state_dict']
        pretrained_dict_t = pretrained_dict_t['state_dict']
        pretrained_dict_s = {k: v for k,v in pretrained_dict_s.items() if 'features' in k}
        pretrained_dict_t = {k: v for k,v in pretrained_dict_t.items() if 'features' in k}
        for k in pretrained_dict_t.keys():
            pretrained_dict_t[k[9:]] = pretrained_dict_t[k]
        for k in pretrained_dict_s.keys():
            pretrained_dict_s[k[9:]] = pretrained_dict_s[k]
        pretrained_dict_s = {k: v for k,v in pretrained_dict_s.items() if k in model_dict_s}
        pretrained_dict_t = {k: v for k,v in pretrained_dict_t.items() if k in model_dict_t}
        
        model_dict_s.update(pretrained_dict_s)
        model_dict_t.update(pretrained_dict_t)
        model.features_s.load_state_dict(model_dict_s)
        model.features_t.load_state_dict(model_dict_t)
        model.cuda()
    print('done!')

    #criterion = torch.nn.BCELoss().cuda()
    criterion = floss().cuda()
    #train_params = list(model.fusion.parameters()) + list(model.decoder.parameters())
    train_params = model.parameters()
    optimizer = torch.optim.Adam(train_params, lr=1e-7)

    for epoch in tqdm(range(epochnow, 31)):
        
        #loss1 = validate(STValLoader, model, criterion)
        #print('epoch%05d, val loss is: %05f' % (epoch, loss1))
        adjust_learning_rate(optimizer, epoch)
        #print('begin training!')
        train(STTrainLoader, model, criterion, optimizer, epoch)
        #loss1 = validate(STValLoader, model, criterion)
        checkpoint_name = "%05d_%s" % (epoch, "fusion3d_bn_floss_checkpoint.pth.tar")
        save_checkpoint({'epoch': epoch, 'arch': 'rgb', 'state_dict': model.state_dict(),},
                                checkpoint_name, save_path)
'''