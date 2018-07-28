import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage import io
import math
import matplotlib.pyplot as plt
import os


def var_to_image(var):
    ten = var.data.cpu()
    if ten.dim() == 4:
        ten = ten[0,:,:,:].squeeze()
    if ten.dim() == 3:
        ten = ten.mul(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
        ten = ten.add(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1))
        ten = ten.numpy()
        ten = ten.transpose((1,2,0))
        return ten
    elif ten.dim() == 2:
        return ten.numpy()
    else:
        print('warning: input variable is invalid to transfer to image')
        return np.zeros(224,224)

class AverageMeter(object):
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


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    elif h is None:
        return None
    else:
        return tuple(repackage_hidden(v) for v in h)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
} #last max pooling removed 

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

def make_layers_e(cfg, in_channels, batch_norm=True):
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if i == 16:
                    layers += [conv2d, nn.BatchNorm2d(v)]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def change_key_names(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    for layer_key in old_params.keys():
        if layer_count < 25:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                rgb_weight_mean = torch.mean(rgb_weight, dim=1, keepdim=True)
                flow_weight = rgb_weight_mean.repeat(1,in_channels,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                print(layer_key, new_params[layer_key].size())
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                print(layer_key, new_params[layer_key].size())
    return new_params

def computeAAEAUC(output, target):
    aae = []
    auc = []
    gp = []
    if output.ndim == 3:
        for batch in range(output.shape[0]):
            out_sq = output[batch,:,:].squeeze()
            tar_sq = target[batch,:,:].squeeze()
            predicted = ndimage.measurements.center_of_mass(out_sq)
            (i,j) = np.unravel_index(tar_sq.argmax(), tar_sq.shape)
            gp.append([i,j])
            d = 112/math.tan(math.pi/6)
            r1 = np.array([predicted[0], predicted[1], d])
            r2 = np.array([i, j, d])
            angle = math.atan2(np.linalg.norm(np.cross(r1,r2)), np.dot(r1,r2))
            aae.append(math.degrees(angle))

            z = np.zeros((224,224))
            z[int(predicted[0])][int(predicted[1])] = 1
            z = ndimage.filters.gaussian_filter(z, 14)
            z = z - np.min(z)
            z = z / np.max(z)
            atgt = z[i][j]
            fpbool = z > atgt
            auc1 = 1 - float(fpbool.sum())/output.shape[2]/output.shape[1]
            auc.append(auc1)
        return np.mean(aae), np.mean(auc), gp
    else:
        predicted = ndimage.measurements.center_of_mass(output)
        (i,j) = np.unravel_index(target.argmax(), target.shape)
        d = 112/math.tan(math.pi/6)
        r1 = np.array([predicted[0], predicted[1], d])
        r2 = np.array([i, j, d])
        angle = math.atan2(np.linalg.norm(np.cross(r1,r2)), np.dot(r1,r2))
        aae = math.degrees(angle)

        z = np.zeros((224,224))
        z[int(predicted[0])][int(predicted[1])] = 1
        z = ndimage.filters.gaussian_filter(z, 14)
        z = z - np.min(z)
        z = z / np.max(z)
        atgt = z[i][j]
        fpbool = z > atgt
        auc = (1 - float(fpbool.sum())/(output.shape[0]*output.shape[1]))
        return aae, auc, [[i,j]]

def plot_loss(train_loss, test_loss, save_path):
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.savefig(save_path)
    plt.close()