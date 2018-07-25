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

from floss import floss
from STdatas import STTrainData, STValData
from utils import *
from LSTMnet import lstmnet
from fusion_st3d import VGG_st_3dfuse
from late_fusion import late_fusion
#from lateDataset import lateDatasetTrain, lateDatasetVal

batch_size = 10
hook_name = 'features_s'

global features_blobs
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output)
    #features_blobs = output.data.cpu().numpy()

'''
class st_3dfuse(nn.Module):
    def __init__(self, features_s, features_t):
        super(st_3dfuse, self).__init__()
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

    def forward(self, x_s, x_t, prev, i):
        x_s = self.features_s(x_s)
        x_t = self.features_t(x_t)
        x_su = self.relu(x_s.unsqueeze(2)) # add dimension D
        x_tu = x_t.unsqueeze(2)
        x_fused = torch.cat((x_su, x_tu), 2)
        x_fused = self.fusion(x_fused)
        x_fused = self.pool3d(x_fused)
        x_fused = x_fused.squeeze_(2)
        x_fused = self.bn(x_fused)
        x = x_fused * prev
        x = self.relu(x_fused)
        x = self.decoder(x_fused)
        x = self.final(x)

        return x



class lstmnet(nn.Module):
    def __init__(self, num_channel=512, num_layer=3, batch_size=16):
        super(lstmnet, self).__init__()
        self.lstm = nn.LSTM(num_channel, num_channel, num_layer)
        self.tanh = nn.Tanh()
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.batch_size = batch_size


    def forward(self, input, hidden):
        # this hidden should be (h, c)
        input = self.tanh(input)
        if hidden is None:
            ihidden = Variable(torch.zeros(self.num_layer, self.batch_size, self.num_channel)).cuda(async = True)
            icell = Variable(torch.zeros(self.num_layer, self.batch_size, self.num_channel)).cuda(async = True)
            inithidden = (ihidden, icell)
            out, hidden = self.lstm(input, inithidden)
        else:
            out, hidden = self.lstm(input, hidden)
        return (out, hidden)
'''

def crop_feature(feature, maxind, size):
    #maxind is gaze point
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        fmax = np.array(maxind[b])
        fmax = fmax / 16  #downsize from 224 to 14
        fmax = np.clip(fmax, size/2, H-int(math.ceil(size/2.0)))
        cfeature = feature[b,:,(fmax[0]-size/2):(fmax[0]+int(math.ceil(size/2.0))),(fmax[1]-size/2):(fmax[1]+int(math.ceil(size/2.0)))]
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

'''
def computeAAEAUC(output, target):
    aae = []
    auc = []
    gp = []
    for batch in range(output.size(0)):
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
        auc.append(1 - float(fpbool.sum())/(output.shape[0]*output.shape[1]))
    return np.mean(aae), np.mean(auc), gp
'''

def train(epoch, st_loader, model, modelw, criterion, optimizer, use_w = False, val = False):
    global features_blobs
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    if val:
        model.eval()
    else:
        model.train()
    modelw.eval()
    hidden = None
    downsample = nn.AvgPool2d(16)
    feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
    currname = None
    for i, sample in enumerate(st_loader):
        if sample['flowmean'].size(0) != batch_size:
            continue
        #reset hidden state only when a video is over
        if currname is None:
            currname = sample['imname'][-1][:-14]
        else:
            if sample['imname'][-1][:-14] != currname:
                hidden = None
                feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
                currname = sample['imname'][-1][:-14]

        flowmean = sample['flowmean']
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        flowmean = flowmean.float().cuda(async=True)
        input_s = input_s.float().cuda(async=True)
        input_t = input_t.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var_s = Variable(input_s)
        input_var_t = Variable(input_t)
        target_var = Variable(target)
        
        features_blobs = []
        output = model(input_var_s, input_var_t)
        feature_fusion = features_blobs[0]

        target_var = target_var.view(output.size())
        loss = criterion(output, target_var)
        if not val:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.update(loss.data[0], input_s.size(0))
        outim = output.cpu().data.numpy().squeeze()
        targetim = target_var.cpu().data.numpy().squeeze()
        aae1, auc1, pred_gp = computeAAEAUC(outim,targetim)
        auc.update(auc1)
        aae.update(aae1)

        if not use_w:
            feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
        else:
            #feature_fusion = Variable(feature_fusion.data)
            cfeature = crop_feature(feature_fusion, pred_gp, 5)
            chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
            chn_weight = torch.mean(chn_weight, 2)
            #print chn_weight.size()   #(batch_size,512)
            chn_weight = torch.cat((chn_weight.data, flowmean), 1) #should be (batch_size, 513)
            chn_weight = chn_weight.unsqueeze(1)  #(seq_len, batch, input_size)
            chn_weight = Variable(chn_weight).cuda(async=True)
            hidden = repackage_hidden(hidden)
            pred_chn_weight, hidden = modelw(chn_weight, hidden)  #pred size (seq_len, batch, output_size) ie (batch_size, 1, 512)
            pred_chn_weight = pred_chn_weight.squeeze()
            feature_fusion = (pred_chn_weight+1)/2  #turn to range(0,1)
            feature_fusion = feature_fusion.view(batch_size, 512, 1, 1)

        if (i+1)%5000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t''AUCAAE {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(st_loader)+1, auc = auc, loss= losses, aae=aae))
    return losses.avg, auc.avg, aae.avg

def visw(st_loader):
    from matplotlib import pyplot as plt
    for i,sample in enumerate(st_loader):
        inp = sample['input'].squeeze()   #(10, 513)
        target = sample['gt'].squeeze()    #(10, 512)
        inp = inp[0,:].cpu().numpy()
        target = target[0,:].cpu().numpy()
        plt.plot(inp)
        plt.show()
        plt.plot(target)
        plt.show()


def trainw(epoch, st_loader, modelw, criterion, optimizer, val=False, verbose=False):
    losses = AverageMeter()
    if not val:
        modelw.train()
    else:
        modelw.eval()
    hidden = None
    feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
    currname = None
    tanh = nn.Tanh()
    relu = nn.ReLU()
    pred_chn_weight = None
    for i, sample in enumerate(st_loader):
        #reset hidden state only when a video is over
        same = sample['same']
        if int(same) == 0:
            hidden = None

        inp = sample['input'].unsqueeze(0)   #(1, 1, 512)
        target = sample['gt'].unsqueeze(0)    #(1,1, 512)

        inp = Variable(inp)
        target_var = Variable(target)

        if pred_chn_weight is not None:
            pred_chn_weight = pred_chn_weight.squeeze()
            loss = criterion(pred_chn_weight, tanh(target_var))
            if not val:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.update(loss.data[0])

        if (i+1)% 2000== 0:
            if verbose:
                print torch.min(pred_chn_weight), torch.max(pred_chn_weight), torch.mean(pred_chn_weight)
                print torch.min(tanh(target_var)), torch.max(tanh(target_var)), torch.mean(tanh(target_var))
            #raw_input('press enter to continue...')
            torch.save({'p':pred_chn_weight, 't':tanh(target_var)}, 'seeletm%03d.pth.tar'%i)
            print('Epoch: [{0}][{1}/{2}]\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(st_loader)+1,  loss= losses))

        hidden = repackage_hidden(hidden)
        pred_chn_weight, hidden = modelw(inp, hidden)
        
    return losses.avg


def vis_features(st_loader, model, modelw, savefolder):
    global features_blobs
    model.eval()
    modelw.eval()
    feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
    pred_chn_weight = None
    downsample = nn.AvgPool2d(16)
    hidden = None
    #sm = nn.Softmax(1)
    for i, sample in enumerate(st_loader):
        if i<100:
            continue
        if i>1000:
            return
        currname = sample['imname'][-1][:-4]
        flowmean = sample['flowmean']
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        flowmean = flowmean.float().cuda(async=True)
        input_s = input_s.float().cuda(async=True)
        input_t = input_t.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var_s = Variable(input_s)
        input_var_t = Variable(input_t)
        target_var = Variable(target)

        target_flat = downsample(target_var).view(target_var.size(0), target_var.size(1), -1)
        _, maxind = torch.max(target_flat, 2)
        feature_fusion = Variable(torch.ones(batch_size,512,1,1)).cuda()
        features_blobs = []
        _ = model(input_var_s, input_var_t, feature_fusion, i)
        feature_fusion = features_blobs[0]

        cfeature = crop_feature_var(feature_fusion, maxind, 5)
        chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
        chn_weight = torch.mean(chn_weight, 2)
        #print torch.mean(chn_weight), torch.max(chn_weight), torch.min(chn_weight)
        #raw_input()
        #chn_weight = sm(chn_weight)

        weighted_feature = feature_fusion * chn_weight.view(batch_size, 512, 1, 1)
        weighted_feature = torch.sum(weighted_feature, 1)
        weighted_feature = weighted_feature[0,:,:].data.cpu().numpy()
        weighted_feature = weighted_feature - np.amin(weighted_feature)
        weighted_feature = weighted_feature / np.amax(weighted_feature)
        weighted_feature = np.uint8(255*weighted_feature)
        weighted_feature = cv2.resize(weighted_feature, (224,224))
        img = cv2.imread('gtea_images/' + sample['imname'][0])
        heatmap = cv2.applyColorMap(weighted_feature, cv2.COLORMAP_JET)
        result = heatmap*0.3 + img*0.5
        cv2.imwrite(savefolder + 'gt_' + sample['imname'][0], result)

        weighted_feature = feature_fusion
        weighted_feature = torch.sum(weighted_feature, 1)
        weighted_feature = weighted_feature[0,:,:].data.cpu().numpy()
        weighted_feature = weighted_feature - np.amin(weighted_feature)
        weighted_feature = weighted_feature / np.amax(weighted_feature)
        weighted_feature = np.uint8(255*weighted_feature)
        weighted_feature = cv2.resize(weighted_feature, (224,224))
        img = cv2.imread('gtea_images/' + sample['imname'][0])
        heatmap = cv2.applyColorMap(weighted_feature, cv2.COLORMAP_JET)
        result = heatmap*0.3 + img*0.5
        cv2.imwrite(savefolder + 'noweight_' + sample['imname'][0], result)

        if pred_chn_weight is not None:
            #pred_chn_weight = sm(pred_chn_weight)
            weighted_feature = feature_fusion * pred_chn_weight.view(batch_size, 512, 1, 1)
            weighted_feature = torch.sum(weighted_feature, 1)
            weighted_feature = weighted_feature[0,:,:].data.cpu().numpy()
            weighted_feature = weighted_feature - np.amin(weighted_feature)
            weighted_feature = weighted_feature / np.amax(weighted_feature)
            weighted_feature = np.uint8(255*weighted_feature)
            weighted_feature = cv2.resize(weighted_feature, (224,224))

            img = cv2.imread('gtea_images/' + sample['imname'][0])
            heatmap = cv2.applyColorMap(weighted_feature, cv2.COLORMAP_JET)
            result = heatmap*0.3 + img*0.5
            cv2.imwrite(savefolder + 'pred_' + sample['imname'][0], result)
        gaze = target_var.data.cpu().numpy()
        gaze = gaze[0,:,:,:].squeeze()
        io.imsave(savefolder + 'gaze_' + sample['imname'][0], gaze)


        #print chn_weight.size()   #(batch_size,512)
        chn_weight = torch.cat((chn_weight.data, flowmean), 1) #should be (batch_size, 513)
        chn_weight = chn_weight.unsqueeze(1)  #(seq_len, batch, input_size)
        chn_weight = Variable(chn_weight).cuda(async=True)
        hidden = repackage_hidden(hidden)
        pred_chn_weight, hidden = modelw(chn_weight, hidden)  #pred size (seq_len, batch, output_size) ie (batch_size, 1, 512)
        pred_chn_weight = pred_chn_weight.squeeze()
        feature_fusion = (pred_chn_weight+1)/2  #turn to range(0,1)
        feature_fusion = feature_fusion.view(batch_size, 512, 1, 1)

def get_weighted(chn_weight, feature):
    #chn_weight (512), feature(1,512,14,14)
    chn_weight = chn_weight.view(1,512,1,1)
    feature = feature * chn_weight
    feature = torch.sum(feature, 1)
    feature = feature - torch.min(feature)
    feature = feature / torch.max(feature)
    #feature = feature - torch.mean(feature)
    return feature

def extract_late(epoch, st_loader, model, modelw):
    global features_blobs
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    aucm = AverageMeter()
    aaem = AverageMeter()
    auc2 = AverageMeter()
    aae2 = AverageMeter()
    model.eval()
    modelw.eval()
    hidden = None
    ds = nn.AvgPool2d(4, stride=4)
    currname = None
    for i, sample in tqdm(enumerate(st_loader)):
        currname = sample['imname'][0]
        fixsac = sample['fixsac']
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        input_s = input_s.float().cuda(async=True)
        input_t = input_t.float().cuda(async=True)
        target = target.float().cuda(async=True)
        input_var_s = Variable(input_s)
        input_var_t = Variable(input_t)
        target_var = Variable(target) #(1,1,224,224)
        features_blobs = []
        output = model(input_var_s, input_var_t)  #(1,1,224,224)
        feature_s = features_blobs[0]  #(1,512,14,14)

        outim = output.cpu().data.numpy().squeeze() #(224,224)
        targetim = target_var.cpu().data.numpy().squeeze() #(224,224)
        outim = np.uint8(255*outim)
        cv2.imwrite('gtea3_pred/'+currname, outim)

        aae1, auc1, pred_gp = computeAAEAUC(outim,targetim)
        #aucm.update(auc1)
        #aaem.update(aae1)

        cfeature = crop_feature(feature_s, pred_gp, 3) #(1,512,3,3)
        cfeature = cfeature.contiguous()
        chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
        chn_weight = torch.mean(chn_weight, 2)  #(1,512)
        if int(fixsac) == 1:
            feat = get_weighted(chn_weight, feature_s)
        else:
            hidden = repackage_hidden(hidden)
            chn_weight, hidden = modelw(chn_weight.unsqueeze(0), hidden)
            chn_weight = chn_weight.squeeze(0)
            feat = get_weighted(chn_weight, feature_s)
        feat = feat.cpu().data.numpy().squeeze()
        feat = np.uint8(255*feat)
        feat = cv2.resize(feat, (224,224))
        cv2.imwrite('gtea3_feat/'+currname, feat)


def train_late(epoch, loader, model, criterion, optimizer, val=False):
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    ds = nn.AvgPool2d(4, stride=4)
    for i,sample in enumerate(loader):
        im = sample['im']
        gt = sample['gt']
        feat = sample['feat']
        im = im.float().cuda(async = True)
        gt = gt.float().cuda(async = True)
        feat = feat.float().cuda(async = True)
        im = Variable(im)
        gt = Variable(gt)
        #gt = ds(gt)
        feat = Variable(feat)
        out = model(feat, im)
        loss = criterion(out, gt)
        outim = out.cpu().data.numpy().squeeze()
        targetim = gt.cpu().data.numpy().squeeze()
        aae1, auc1, _ = computeAAEAUC(outim,targetim)
        auc.update(auc1)
        aae.update(aae1)
        losses.update(loss.data[0])
        if not val:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (i+1)%3060 == 0 or i == 0:
            print('Epoch: [{0}][{1}/{2}]\t''AUCAAE_late {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(loader)+1, auc = auc, loss= losses, aae=aae,))

    return losses.avg, auc.avg, aae.avg

def adjust_learning_rate(optimizer, epoch, lr):

    #lr = lr / 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    load_from = 4
    trained_model = 'savefusion/%05d_fusion3d_bn_floss_checkpoint.pth.tar'%load_from
    #trained_model = 'savelstm/3layerall/0net.pth.tar'
    print('building pretrained model from epoch %02d...'%load_from)
    model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
    pretrained_dict = torch.load(trained_model)
    model_dict = model.state_dict()
    #model_dict.update(pretrained_dict)
    model_dict.update(pretrained_dict['state_dict'])
    model.load_state_dict(model_dict, strict=False)
    
    model.cuda()

    model._modules.get(hook_name).register_forward_hook(hook_feature)
    criterion = floss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    modelw = lstmnet()
    modelw.cuda()
    criterionw = nn.MSELoss().cuda()
    optimizerw = torch.optim.Adam(modelw.parameters(), lr=1e-4)

    load_lstm = False

    if load_lstm:
        trained_model = 'savelstm/3layerall/best_train3_lstmnet.pth.tar'
        pretrained_dict = torch.load(trained_model)
        model_dict = modelw.state_dict()
        model_dict.update(pretrained_dict)
        modelw.load_state_dict(model_dict)

    load_late = False
    model_late = late_fusion()
    model_late.cuda()
    if load_late:
        trained_model = 'savelate/best_train.pth.tar'
        pretrained_dict = torch.load(trained_model)
        model_dict = model_late.state_dict()
        model_dict.update(pretrained_dict)
        model_late.load_state_dict(model_dict)
        del pretrained_dict

    optimizer_late = torch.optim.Adam(model_late.parameters(), lr=1e-4)
    criterion = floss().cuda()
    print('init done!')
    
    #vis_features(STTrainLoader, model, modelw, 'savelstm/3layerall/vistrainrelu/')


    from wdatas import wTrainData, wValData
    wTrainLoader = DataLoader(dataset=wTrainData, batch_size=1, shuffle=False, num_workers=0)
    wValLoader = DataLoader(dataset=wValData, batch_size=1, shuffle=False, num_workers=0)
    train_lstm = True
    if train_lstm:
        print('begin training lstm....')
        #trainw(0, wTrainLoader, modelw, criterionw, optimizerw, val=True)
        #trainw(0, wTrainLoader, modelw, criterionw, optimizerw)
        prev = 999
        prevt = 999
        for epoch in range(120):
            #lr = raw_input('please input lr:')
            #lr = float(lr)
            #adjust_learning_rate(optimizerw, epoch, lr)
            l = trainw(epoch, wTrainLoader, modelw, criterionw, optimizerw, verbose = False)
            print ('---------train loss: %f-----------'%l)
            if l < prevt:
                torch.save(modelw.state_dict(), 'savelstm/3layerall/best_train3_lstmnet.pth.tar')
            l = trainw(0, wValLoader, modelw, criterionw, optimizerw, val=True, verbose = False)
            print ('----------val loss: %f-------------'%l)
            if epoch == -4:
                adjust_learning_rate(optimizerw, epoch, 1e-5)
            if l<prev:
                prev=l
                print epoch
                torch.save(modelw.state_dict(), 'savelstm/3layerall/best_val3_lstmnet.pth.tar')
        print('lstm training finished!')

    extract_late(0, DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True), model, modelw)
    #extract_late(0, DataLoader(dataset=STTrainData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True), model, modelw)

    #STTrainLoader = DataLoader(dataset=STTrainData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    #STValLoader = DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    '''
    del model
    del modelw
    from lateDataset import lateDatasetTrain, lateDatasetVal
    train_loader = DataLoader(dataset = lateDatasetTrain, batch_size = 32, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset = lateDatasetVal, batch_size = 32, shuffle=False, num_workers=1, pin_memory=True)
    trainprev = 999
    valprev = 999
    for epoch in range(100):
        print ('begin training model....')
        loss, auc, aae = train_late(epoch, train_loader, model_late, criterion, optimizer_late, val = False)
        print('training, auc is %5f, aae is %5f'%(auc, aae))
        if loss < trainprev:
            torch.save({'state_dict': model_late.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, 'savelate/best_train_3.pth.tar')
            trainprev = loss
        print('begin validation...')
        loss, auc, aae = train_late(epoch, val_loader, model_late, criterion, optimizer_late, val = True)
        #extract_late(epoch, STTrainLoader, model, modelw, model_late, criterion, optimizer_late, val = False)
        #extract_late(epoch, STValLoader, model, modelw, model_late, criterion, optimizer_late, val = False)
        print('val, auc is %5f, aae is %5f'%(auc, aae))
        if loss < valprev:
            torch.save({'state_dict': model_late.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, 'savelate/best_val_3.pth.tar')
            valprev = loss
    '''