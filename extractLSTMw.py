import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from scipy import ndimage
from skimage import io
import math
from tqdm import tqdm
import cv2

from data.STdatas import STDataset
from models.LSTMnet import lstmnet
from utils import *

"""
Extract attention weights, save them for training LSTM and later use.

"""

class st_extract(nn.Module):
    def __init__(self, features_s):
        super(st_extract, self).__init__()
        self.features_s = features_s

    def forward(self, x_s):
        x = self.features_s(x_s)
        return x

def crop_feature_align(feature, maxind, size):
    size *= 16
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        ind = maxind[b].item()
        fmax = np.unravel_index(ind, (H,W))
        fmax = np.clip(fmax, size//2, (H-size//2))
        cfeature = feature[b,:,(fmax[0]-size//2):(fmax[0]+size//2),(fmax[1]-size//2):(fmax[1]+size//2)]
        cfeature = cfeature.unsqueeze(0)
        if b==0:
            res = cfeature
        else:
            res = torch.cat((res, cfeature),0)
    return res

def crop_feature_var(feature, maxind, size):
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        ind = maxind[b].item()
        fmax = np.unravel_index(ind, (H,W))
        fmax = np.clip(fmax, size/2, H-int(math.ceil(size/2.0)))
        cfeature = feature[b,:,int(fmax[0]-size/2):int(fmax[0]+int(math.ceil(size/2.0))),int(fmax[1]-size/2):int(fmax[1]+int(math.ceil(size/2.0)))]
        cfeature = cfeature.unsqueeze(0)
        if b==0:
            res = cfeature

def extractw(loader, model, savepath, crop_size=3, device='cuda:0', align=False):
    print('extracting lstm training data...')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fixflag = False
    fix2flag = False
    downsample = nn.AvgPool2d(16)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i == 0:
                print ('0%')
            if i == 1:
                print ('1%')
            if i == len(loader)//2:
                print ('50%')
            fixsac = sample['fixsac']
            if fixflag == False and float(fixsac) == 1.0:
                fixflag=True
            elif fixflag == False and float(fixsac) == 0.0:
                continue
            elif fixflag == True and fix2flag == False and float(fixsac) == 1.0:
                fix2flag = True
                currname = sample['imname'][0][:-4]
                inp = sample['image']  #(1,3,224,224)
                inp = inp.float().to(device)
                target = sample['gt'] #(1,1,224,224)
                target = target.float().to(device)
                if align:
                    target_flat = target.view(target.size(0), target.size(1), -1)
                else:
                    target_flat = downsample(target).view(target.size(0), target.size(1), -1) #(1,1,196)
                _, maxind = torch.max(target_flat, 2) #(1,1)

                out = model(inp) #(1,512,14,14)
                if align:
                    out = nn.functional.upsample_bilinear(out, scale_factor = 16)
                    cfeature = crop_feature_align(out, maxind, crop_size) #(1,512,h,w)
                else:
                    cfeature = crop_feature_var(out, maxind, crop_size) #(1,512,h,w)
                cfeature = cfeature.contiguous()
                chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
                chn_weight = torch.mean(chn_weight, 2) #(1,512)
                chn_weight = chn_weight.data.squeeze() #(512)
                torch.save(chn_weight, os.path.join(savepath , 'fix_'+currname+'.pth.tar'))
            elif fixflag == True and fix2flag ==True and float(fixsac) == 1.0:
                continue
            elif fixflag == True and fix2flag == True and float(fixsac) == 0.0:
                fixflag = False
                fix2flag = False
            else:
                raise RuntimeError('fixation is not processed.')
    print('done')

        
def extract_LSTM_training_data(save_path='../512w', trained_model='save/best_fusion.pth.tar', device='0', crop_size=3, traindata=None, valdata=None, align=False):
    
    batch_size = 1
    device = 'cuda:'+ device
    
    STTrainLoader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    STValLoader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    model = st_extract(make_layers(cfg['D'], 3))
    model.to(device)
    pretrained_dict = torch.load(trained_model)
    pretrained_dict = pretrained_dict['state_dict']
    load_dict = {k:v for k,v in pretrained_dict.items() if 'features_s' in k}
    model_dict = model.state_dict()
    for k in load_dict.keys():
        if k not in model_dict.keys():
            del load_dict[k]
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    del pretrained_dict, load_dict


    extractw(STTrainLoader, model, os.path.join(save_path, 'train'), crop_size, device, align)
    extractw(STValLoader, model, os.path.join(save_path, 'test'), crop_size, device, align)
    print('Attention weight for training LSTMnet successfully extracted.')