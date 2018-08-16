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
from models.model_SP import model_SP
from utils import *

class SP():
    def __init__(self, lr=1e-7, loss_save='loss_SP.png', save_name='best_fusion.pth.tar', save_path='save', loss_function='f',\
        num_epoch=10, batch_size=10, device='0', resume=1, pretrained_spatial='save/04_spatial.pth.tar', pretrained_temporal='save/03_temporal.pth.tar'):
        self.lr = lr
        self.loss_save = loss_save
        self.save_name = save_name
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.loss_function = loss_function
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.device = torch.device('cuda:'+device)
        self.pretrained_spatial = pretrained_spatial
        self.pretrained_temporal = pretrained_temporal
        self.STTrainLoader = DataLoader(dataset=STTrainData, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        self.STValLoader = DataLoader(dataset=STValData, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        # 2: resume from fusion
        # 0: from vgg
        # 1: resume from separately pretrained models
        in_channels = 20
        if resume == 2:
            self.model = model_SP(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
            trained_model = os.path.join(save_path, save_name)
            pretrained_dict = torch.load(trained_model)
            self.epochnow = pretrained_dict['epoch']
            pretrained_dict = pretrained_dict['state_dict']
            self.model.to(self.device)
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        elif resume == 0:
            self.epochnow = 0
            self.model = model_SP(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
            pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')

            model_dict_s = self.model.features_s.state_dict()
            model_dict_t = self.model.features_t.state_dict()
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
            self.model.features_s.load_state_dict(model_dict_s)
            self.model.features_t.load_state_dict(model_dict_t)
            self.model.to(self.device)
        else:
            self.epochnow = 0
            self.model = model_SP(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
            pretrained_dict_s = torch.load(self.pretrained_spatial)
            pretrained_dict_t = torch.load(self.pretrained_temporal)
            model_dict_s = self.model.features_s.state_dict()
            model_dict_t = self.model.features_t.state_dict()
            pretrained_dict_s = pretrained_dict_s['state_dict']
            pretrained_dict_t = pretrained_dict_t['state_dict']
            pretrained_dict_s = {k: v for k,v in pretrained_dict_s.items() if 'features' in k}
            pretrained_dict_t = {k: v for k,v in pretrained_dict_t.items() if 'features' in k}
            new_pretrained_dict_t = {}
            new_pretrained_dict_s = {}
            for k in pretrained_dict_t.keys():
                new_pretrained_dict_t[k[9:]] = pretrained_dict_t[k]
            for k in pretrained_dict_s.keys():
                new_pretrained_dict_s[k[9:]] = pretrained_dict_s[k]
            new_pretrained_dict_s = {k: v for k,v in pretrained_dict_s.items() if k in model_dict_s}
            new_pretrained_dict_t = {k: v for k,v in pretrained_dict_t.items() if k in model_dict_t}
            
            model_dict_s.update(new_pretrained_dict_s)
            model_dict_t.update(new_pretrained_dict_t)
            self.model.features_s.load_state_dict(model_dict_s)
            self.model.features_t.load_state_dict(model_dict_t)
            self.model.to(self.device)
        if loss_function != 'f':
            self.criterion = torch.nn.BCELoss().to(self.device)
        else:
            self.criterion = floss().to(self.device)

        # train params may be change according to resume state
        self.optimizer = torch.optim.Adam([{'params': list(self.model.fusion.parameters())+list(self.model.bn.parameters())+list(self.model.decoder.parameters()),}
                                 ], lr=self.lr)
        print('SP module init done!')

    def trainSP(self):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        self.optimizer.zero_grad()
        loss_mini_batch = 0.0
        for i, sample in tqdm(enumerate(self.STTrainLoader)):
            input_s = sample['image']
            target = sample['gt']
            input_t = sample['flow']
            input_s = input_s.float().to(self.device)
            input_t = input_t.float().to(self.device)
            target = target.float().to(self.device)
            output = self.model(input_s, input_t)
            target = target.view(output.size())
            loss = self.criterion(output, target)
            loss_mini_batch += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_time.update(time.time() - end)
            losses.update(loss.item(), input_s.size(0))
            end = time.time()
            loss_mini_batch = 0
            if (i+1)%1000 == 0:
                print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    self.epochnow, i+1, len(st_loader)+1, batch_time = batch_time, loss= losses))
        return losses.avg

    def testSP(self):
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        auc = AverageMeter()
        aae = AverageMeter()
        end = time.time()
        with torch.no_grad():
            for i, sample in tqdm(enumerate(self.STValLoader)):
                input_s = sample['image']
                target = sample['gt']
                input_t = sample['flow']
                input_s = input_s.float().to(self.device)
                input_t = input_t.float().to(self.device)
                target = target.float().to(self.device)
                output = self.model(input_s, input_t)
                target = target.view(output.size())
                loss = self.criterion(output, target)
                losses.update(loss.item(), input_s.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                outim = output.cpu().data.numpy().squeeze()
                targetim = target.cpu().data.numpy().squeeze()
                aae1, auc1, _ = computeAAEAUC(outim, targetim)
                auc.update(auc1)
                aae.update(aae1)
                '''
                if i == 836:  #inception of results, actually completely useless
                    outim = output.data.cpu().numpy()
                    outim = outim[0,:,:,:].squeeze()
                    io.imsave(os.path.join(self.save_path,'fusion_test_%05d.jpg'%i),outim)
                    if not os.path.exists(os.path.join(self.save_path,'targetfusion_%05d.jpg')):
                        targetim = target.data.cpu().numpy()
                        targetim = targetim[0,:,:,:].squeeze()
                        io.imsave(os.path.join(self.save_path,'targetfusion_%05d.jpg'%i), targetim)
                '''
                if i % 1000 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(st_loader), batch_time=batch_time, loss=losses,))
        print ('AUC: {0}\t AAE: {1}'.format(auc.avg, aae.avg))
        return losses.avg, auc.avg, aae.avg

    def train(self):
        train_loss = []
        val_loss = []
        best_loss = 100

        for epoch in range(self.epochnow, self.num_epoch):
            self.epochnow = epoch
            loss1 = self.trainSP()
            train_loss.append(loss1)
            loss1, auc1, aae1 = self.testSP()
            val_loss.append(loss1)
            plot_loss(train_loss, val_loss, os.path.join(self.save_path, self.loss_save))
            checkpoint_name = self.save_name
            if loss1 < best_loss:
                best_loss = loss1
                save_checkpoint({'epoch': epoch, 'arch': 'SP', 'state_dict': self.model.state_dict(),'optimizer':self.optimizer.state_dict(), 'auc': auc1, 'aae': aae1},
                                    checkpoint_name, self.save_path)
