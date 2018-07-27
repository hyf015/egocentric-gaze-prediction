import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.model_zoo as model_zoo
import os
import time
import numpy as np
from skimage import io, transform
import math
from tqdm import tqdm

#problem for now: no normalization, model doesn't learn, batch size =1 and has to be set in train() and validate()

global lr, listTrainFiles, listGtFiles, listValFiles, listValGtFiles
lr = 1e-7
imgPath = 'gtea_images'
gtPath = 'gtea_gts'
listTrainFiles = [k for k in os.listdir(imgPath) if 'Alireza' not in k]
listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
listValFiles = [k for k in os.listdir(imgPath) if 'Alireza' in k]
listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
listTrainFiles.sort()
listValFiles.sort()
listGtFiles.sort()
listValGtFiles.sort()

class SpatialDatasetTrain(Dataset):
    def __init__(self, imgPath, gtPath, transform = None):
        global listTrainFiles
        self.listTrainFiles = listTrainFiles
        #self.listValFiles = [k.split('/')[-1].split(.) for k in glob.glob(os.path.join(imgPath)) if 'Alireza' in k]
        #self.listTestFiles = [k.split('/')[-1].split(.) for k in glob.glob(os.path.join(imgPath)) if 'Shaghayegh' in k]
        #self.listTrainFiles.sort()
        #self.listValFiles.sort()
        #self.listTestFiles.sort()
        self.listGtFiles = listGtFiles
        #self.listGtFiles.sort()
        self.transform = transform
        self.imgPath = imgPath
        self.gtPath = gtPath
    
    def __len__(self):
        return len(self.listGtFiles)

    def __getitem__(self, index):
        im = io.imread(self.imgPath + '/' + self.listTrainFiles[index])
        gt = io.imread(self.gtPath + '/' + self.listGtFiles[index])
        im = im.transpose((2,0,1))
        im = torch.from_numpy(im)
        im = im.float().div(255)
        im = im.sub_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1)).div_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
        gt = torch.from_numpy(gt)
        gt = gt.float().div(255)
        sample = {'image': im, 'gt': gt}
        return sample

class SpatialDatasetVal(Dataset):
    def __init__(self, imgPath, gtPath, transform = None):
        global listValFiles, listValGtFiles
        self.listValFiles = listValFiles
        #self.listValFiles.sort()
        self.listGtFiles = listValGtFiles
        #self.listGtFiles.sort()
        self.transform = transform
        self.imgPath = imgPath
        self.gtPath = gtPath

    def __len__(self):
        return len(self.listGtFiles)

    def __getitem__(self, index):
        im = io.imread(self.imgPath + '/' + self.listValFiles[index])
        gt = io.imread(self.gtPath + '/' + self.listGtFiles[index])
        im = im.transpose((2,0,1))
        im = torch.from_numpy(im)
        im = im.float().div(255)
        im = im.sub_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1)).div_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
        gt = torch.from_numpy(gt)
        gt = gt.float().div(255)
        sample = {'image': im, 'gt': gt}
        return sample


SpatialTrainData = SpatialDatasetTrain(imgPath = 'gtea_images', gtPath = 'gtea_gts')
SpatialTrainLoader = DataLoader(dataset=SpatialTrainData, batch_size=30, shuffle=True, num_workers=1, pin_memory=False)

SpatialValData = SpatialDatasetVal(imgPath = 'gtea_images', gtPath = 'gtea_gts', transform = None)
SpatialValLoader = DataLoader(dataset=SpatialValData, batch_size=30, shuffle=False, num_workers=1, pin_memory=False)

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
        for param in self.features.parameters():
    		param.requires_grad = False
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
        y = self.final(x)
        return y
    
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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


def adjust_learning_rate(optimizer, epoch):

    global lr
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        input = sample['image']
        target = sample['gt']
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)
        output = model(input_var)
        #outim = beforesigmoid.data.cpu().numpy()
        #outim = outim[0,:,:,:].squeeze()
        #print outim
        #if (i+1) % 10 == 0:
        	#outim = output.data.cpu().numpy()
        	#outim = outim[0,:,:,:].squeeze()
        #print outim.dot(255).astype(int)
        	#io.imsave('e' + str(epoch) + '_' + str(i) + 'watch.jpg',outim)
        #raw_input()
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
            if (i+1) % 100 ==0:
                print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses))


## need to be done: validate, test loader
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    for i, sample in enumerate(val_loader):
        input = sample['image']
        target = sample['gt']
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var = Variable(input, volatile = True)
        target_var = Variable(target, volatile = True)
        output = model(input_var)
        target_var = target_var.view(output.size())
        if i == 1000:
            #outim = output.view(30,224,224)
            outim = output.data.cpu().numpy()
            outim = outim[0,:,:,:].squeeze()
            io.imsave(str(epoch)+'_1000_test.jpg',outim)
            targetim = target_var.data.cpu().numpy()
            targetim = targetim[0,:,:,:].squeeze()
            print targetim.sum()
            io.imsave('target.jpg', targetim)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses,))
    return losses.avg


# main

save_path = 'save'
resume = 1
# our model
if resume:
	print('building model...')
	model = VGG(make_layers(cfg['D']))
	trained_model = 'save/00003_floss_checkpoint.pth.tar'
	pretrained_dict = torch.load(trained_model)
	pretrained_dict = pretrained_dict['state_dict']
	model_dict = model.state_dict()
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	model.cuda()
	print('done!')
else:
	print('building model...')
	model = VGG(make_layers(cfg['D']))
	pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
	model_dict = model.state_dict()
	pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	model.cuda()
	print('done!')

criterion = torch.nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-7)

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Training loop
for epoch in tqdm(range(30001)):
    #loss1 = validate(SpatialValLoader, model, criterion)
    adjust_learning_rate(optimizer, epoch)
    train(SpatialTrainLoader, model, criterion, optimizer, epoch)
    loss1 = validate(SpatialValLoader, model, criterion, epoch)
    print('epoch%05d, val loss is: %05f' % (epoch, loss1))
    checkpoint_name = "%05d_%s" % (epoch, "checkpoint.pth.tar")
    save_checkpoint({'epoch': epoch, 'arch': 'rgb', 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),},
                            checkpoint_name, save_path)


'''
# code for debug
input = torch.randn(16,3,224,224)
input = input.float().cuda(async=True)
input_var = torch.autograd.Variable(input, volatile=True)
output = model(input_var)
print(model.modules)
print output.size()
'''

















