import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from scipy import ndimage
from skimage import io
import math
from tqdm import tqdm
import cv2

from STdatas import STTrainData, STValData
from LSTMnet import lstmnet

class st_extract(nn.Module):
    def __init__(self, features_s):
        super(st_extract, self).__init__()
        self.features_s = features_s

    def forward(self, x_s):
        x = self.features_s(x_s)
        return x

def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

batch_size = 1

STTrainLoader = DataLoader(dataset=STTrainData, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
STValLoader = DataLoader(dataset=STValData, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
def crop_feature_var(feature, maxind, size):
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        ind = maxind[b].data[0]
        fmax = np.unravel_index(ind, (H,W))
        fmax = np.clip(fmax, size/2, H-int(math.ceil(size/2.0)))
        cfeature = feature[b,:,(fmax[0]-size/2):(fmax[0]+int(math.ceil(size/2.0))),(fmax[1]-size/2):(fmax[1]+int(math.ceil(size/2.0)))]
        cfeature = cfeature.unsqueeze(0)
        if b==0:
            res = cfeature
        else:
            res = torch.cat((res, cfeature),0)
    return res

def extractw(loader, model, savepath):
    model.eval()
    fixflag = False
    fix2flag = False
    downsample = nn.AvgPool2d(16)
    for i, sample in tqdm(enumerate(loader)):
        fixsac = sample['fixsac']
        if fixflag == False and float(fixsac) == 1.0:
            fixflag=True
        elif fixflag == False and float(fixsac) == 0.0:
            continue
        elif fixflag == True and fix2flag == False and float(fixsac) == 1.0:
            fix2flag = True
            currname = sample['imname'][0][:-4]
            inp = sample['image']  #(1,3,224,224)
            inp = inp.float().cuda(async = True)
            inp = Variable(inp)
            target = sample['gt'] #(1,1,224,224)
            target = target.float().cuda(async=True)
            target_var = Variable(target)
            target_flat = downsample(target_var).view(target_var.size(0), target_var.size(1), -1) #(1,1,196)
            _, maxind = torch.max(target_flat, 2) #(1,1)

            out = model(inp) #(1,512,14,14)
            cfeature = crop_feature_var(out, maxind, 3) #(1,512,3,3)
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
            print 'what???????????????????????/'
            print i
            fixflag = False
            fix2flag = False

        

print('building model...')
model = st_extract(make_layers(cfg['D'], 3))
model.cuda()
trained_model = 'savefusion/00004_fusion3d_bn_floss_checkpoint.pth.tar'
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

extractw(STTrainLoader, model, '512ww/train')
extractw(STValLoader, model, '512ww/test')