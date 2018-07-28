import torch
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
from models.SP import VGG_st_3dfuse
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-7, required=False)
parser.add_argument('--loss_save', default='loss_fusion.png', required=False)
parser.add_argument('--save_name', default='best_fusion.pth.tar', required=False)
parser.add_argument('--save_path', default='savefusion', required=False)
parser.add_argument('--loss_function', default='f', required=False)
parser.add_argument('--num_epoch', type=int, default=10, required=False)
parser.add_argument('--device', default='0')
parser.add_argument('--resume', type=int, default=0, help='0 from vgg, 1 from separately pretrained models, 2 from pretrained fusion model.')
args = parser.parse_args()

device = torch.device('cuda:'+args.device)

##############################################################spatialtemporal data loader#######################################################

STTrainLoader = DataLoader(dataset=STTrainData, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

STValLoader = DataLoader(dataset=STValData, batch_size=10, shuffle=False, num_workers=1, pin_memory=True)

##############################################################spatialtemporal data loader#######################################################


def save_checkpoint(state,filename,save_path):
    torch.save(state, os.path.join(save_path, filename))


def train(st_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    for i, sample in enumerate(st_loader):
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        input_s = input_s.float().to(device)
        input_t = input_t.float().to(device)
        target = target.float().to(device)
        output = model(input_s, input_t)
        target = target.view(output.size())
        loss = criterion(output, target)
        loss_mini_batch += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        losses.update(loss.item(), input_s.size(0))
        end = time.time()
        loss_mini_batch = 0
        if (i+1)%1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(st_loader)+1, batch_time = batch_time, loss= losses))
    return losses.avg

def validate(st_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for i, sample in enumerate(st_loader):
            input_s = sample['image']
            target = sample['gt']
            input_t = sample['flow']
            input_s = input_s.float().to(device)
            input_t = input_t.float().to(device)
            target = target.float().to(device)
            output = model(input_s, input_t)
            target = target.view(output.size())
            loss = criterion(output, target)
            losses.update(loss.item(), input_s.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            outim = output.cpu().data.numpy().squeeze()
            targetim = target.cpu().data.numpy().squeeze()
            aae1, auc1, _ = computeAAEAUC(outim, targetim)
            auc.update(auc1)
            aae.update(aae1)
            if i == 836:  #random inception of results, actually completely useless
                outim = output.data.cpu().numpy()
                outim = outim[0,:,:,:].squeeze()
                io.imsave(os.path.join(args.save_path,'fusion_test_%05d.jpg'%i),outim)
                if not os.path.exists(os.path.join(args.save_path,'targetfusion_%05d.jpg')):
                    targetim = target.data.cpu().numpy()
                    targetim = targetim[0,:,:,:].squeeze()
                    io.imsave(os.path.join(args.save_path,'targetfusion_%05d.jpg'%i), targetim)
            if i % 1000 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(st_loader), batch_time=batch_time, loss=losses,))
    print ('AUC: {0}\t AAE: {1}'.format(auc.avg, aae.avg))
    return losses.avg



if __name__ == '__main__':
    lr = args.lr
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('building model...')
    in_channels = 20
    resume = args.resume
    # 2: resume from fusion
    # 0: from vgg
    # 1: resume from separately pretrained models
    if resume == 2:
        model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
        trained_model = os.path.join(save_path,'00004_fusion3d_bn_floss_checkpoint.pth.tar')
        pretrained_dict = torch.load(trained_model)
        epochnow = pretrained_dict['epoch']
        pretrained_dict = pretrained_dict['state_dict']
        model.to(device)
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
        model.to(device)
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
        model.to(device)
    print('done!')

    if args.loss_function != 'f':
        criterion = torch.nn.BCELoss().to(device)
    else:
        criterion = floss().to(device)
    #train_params = list(model.fusion.parameters()) + list(model.decoder.parameters())
    train_params = model.parameters()
    optimizer = torch.optim.Adam(train_params, lr=args.lr)
    train_loss = []
    val_loss = []
    best_loss = 100

    for epoch in tqdm(range(epochnow, args.num_epoch)):
        
        #loss1 = validate(STValLoader, model, criterion)
        #print('epoch%05d, val loss is: %05f' % (epoch, loss1))
        #print('begin training!')
        loss1 = train(STTrainLoader, model, criterion, optimizer, epoch)
        train_loss.append(loss1)
        loss1 = validate(STValLoader, model, criterion)
        val_loss.append(loss1)
        plot_loss(train_loss, val_loss, save_path)
        checkpoint_name = args.save_name
        if loss1 < best_loss:
            best_loss = loss1
            save_checkpoint({'epoch': epoch, 'arch': 'fusion', 'state_dict': model.state_dict(),'optimizer':optimizer.state_dict()},
                                checkpoint_name, save_path)