import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.model_zoo as model_zoo
from scipy import ndimage
import os
import numpy as np
from skimage import io, transform
import math
import time
import collections
from floss import floss
from STdatas import STTrainData, STValData

STTrainLoader = DataLoader(dataset=STTrainData, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

STValLoader = DataLoader(dataset=STValData, batch_size=10, shuffle=False, num_workers=1, pin_memory=True)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 20
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
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

def train(train_loader, model, criterion, optimizer, epoch):
    iter_size = 1
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    for i, sample in enumerate(train_loader):
        input = sample['flow']
        target = sample['gt']
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)
        output = model(input_var)
        target_var = target_var.view(output.size())
        loss = criterion(output, target_var)
        #loss = loss / iter_size
        loss_mini_batch += loss.data[0]
        loss.backward()
        
        if (i+1) % iter_size == 0:
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
    for i, sample in enumerate(val_loader):
        i+=1
        input = sample['flow']
        target = sample['gt']
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = Variable(input, volatile = True)
        target_var = Variable(target, volatile = True)
        output = model(input_var)
        target_var = target_var.view(output.size())

        outim = output.cpu().data.numpy().squeeze()
        targetim = target_var.cpu().data.numpy().squeeze()
        auc1 = computeAUC(outim,targetim)
        aae1 = computeAAE(outim,targetim)
        auc.update(auc1)
        aae.update(aae1)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1000 == 0:
            print ('val loss is: %1.4f' % losses.avg)
    #print ('should not be here')
    return losses.avg, auc.avg, aae.avg

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

# main

save_path = 'saveflow'
# our model
resume = 0
epochold = 0
if resume:
    print('building model...')
    model = VGG(make_layers(cfg['D']))
    trained_model = 'saveflow/00004_bn_floss_checkpoint.pth.tar'
    pretrained_dict = torch.load(trained_model)
    epochold = pretrained_dict['epoch']
    pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    print('done!')
else:
    print('building model...')
    model = VGG(make_layers(cfg['D']))
    in_channels = 20
    pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
    model_dict = model.state_dict()
    new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
    new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    print('done!')

#criterion = torch.nn.BCELoss().cuda()
criterion = floss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Training loop
prevloss = 999
for epoch in range(epochold, 10):
    #adjust_learning_rate(optimizer, epoch)
    train(STTrainLoader, model, criterion, optimizer, epoch)
    if epoch % 1 == 0:
        loss, auc, aae = validate(STValLoader, model, criterion, epoch)
        if prevloss > loss:
            prevloss = loss
            print(loss,auc,aae)
            checkpoint_name = "%05d_%s" % (epoch, "bn_floss_checkpoint.pth.tar")
            save_checkpoint({'epoch': epoch, 'arch': 'flow', 'state_dict': model.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, checkpoint_name, save_path)
















