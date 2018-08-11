import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
from skimage import io
import math
from tqdm import tqdm
import cv2, os

from floss import floss
from utils import *
from AT import AT
from LF import LF
import argparse

print('importing done!')
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-7, required=False, help='lr for Adam')
parser.add_argument('--late_save_img', default='loss_late.png', required=False)
parser.add_argument('--pretrained_model', default='save/best_fusion.pth.tar', required=False)
parser.add_argument('--pretrained_lstm', default=None, required=False)
parser.add_argument('--pretrained_late', default='save/best_late.pth.tar', required=False)
parser.add_argument('--lstm_save_img', default='loss_lstm.png', required=False)
parser.add_argument('--save_lstm', default='best_lstm.pth.tar', required=False)
parser.add_argument('--save_late', default='best_late.pth.tar', required=False)
parser.add_argument('--save_path', default='save', required=False)
parser.add_argument('--loss_function', default='f', required=False)
parser.add_argument('--num_epoch', type=int, default=10, required=False)
parser.add_argument('--num_epoch_lstm', type=int, default=30, required=False)
parser.add_argument('--extract_lstm', action='store_true')
parser.add_argument('--train_lstm', action='store_true')
parser.add_argument('--train_late', action='store_true')
parser.add_argument('--extract_late', action='store_true')
parser.add_argument('--extract_late_pred_folder', default='../new_pred/', required=False)
parser.add_argument('--extract_late_feat_folder', default='../new_feat/', required=False)
parser.add_argument('--device', default='0')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--crop_size', type=int, default=3)
args = parser.parse_args()

device = torch.device('cuda:'+args.device)

batch_size = args.batch_size
hook_name = 'features_s'

global features_blobs
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output)

def crop_feature(feature, maxind, size):
    #maxind is gaze point
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        fmax = np.array(maxind[b])
        fmax = fmax // 16  #downsize from 224 to 14
        fmax = np.clip(fmax, size//2, H-int(math.ceil(size/2.0)))
        cfeature = feature[b,:,(fmax[0]-size//2):(fmax[0]+int(math.ceil(size/2.0))),(fmax[1]-size//2):(fmax[1]+int(math.ceil(size/2.0)))]
        cfeature = cfeature.unsqueeze(0)
        if b==0:
            res = cfeature
        else:
            res = torch.cat((res, cfeature),0)
    return res




def train_late(epoch, loader, model, criterion, optimizer):
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    for i,sample in enumerate(loader):
        im = sample['im']
        gt = sample['gt']
        feat = sample['feat']
        im = im.float().to(device)
        gt = gt.float().to(device)
        feat = feat.float().to(device)
        out = model(feat, im)
        loss = criterion(out, gt)
        outim = out.cpu().data.numpy().squeeze()
        targetim = gt.cpu().data.numpy().squeeze()
        aae1, auc1, _ = computeAAEAUC(outim,targetim)
        auc.update(auc1)
        aae.update(aae1)
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%300 == 0:
            print('Epoch: [{0}][{1}/{2}]\t''AUCAAE_late {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i+1, len(loader)+1, auc = auc, loss= losses, aae=aae,))

    return losses.avg, auc.avg, aae.avg

def val_late(epoch, loader, model, criterion):
    losses = AverageMeter()
    auc = AverageMeter()
    aae = AverageMeter()
    with torch.no_grad():
        for i,sample in enumerate(loader):
            im = sample['im']
            gt = sample['gt']
            feat = sample['feat']
            im = im.float().to(device)
            gt = gt.float().to(device)
            feat = feat.float().to(device)
            out = model(feat, im)
            loss = criterion(out, gt)
            outim = out.cpu().data.numpy().squeeze()
            targetim = gt.cpu().data.numpy().squeeze()
            aae1, auc1, _ = computeAAEAUC(outim,targetim)
            auc.update(auc1)
            aae.update(aae1)
            losses.update(loss.item())
            if (i+1) % 1000 == 0:
                print('Epoch: [{0}][{1}/{2}]\t''AUCAAE_late {auc.avg:.3f} ({aae.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i+1, len(loader)+1, auc = auc, loss= losses, aae=aae,))

    return losses.avg, auc.avg, aae.avg


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    att = AT(pretrained_model =args.pretrained_model, pretrained_lstm = args.pretrained_lstm, extract_lstm = False, \
            crop_size = 3, num_epoch_lstm = 30, lstm_save_img = 'loss_lstm_fortest.png',\
            save_path = 'save', save_name = 'best_lstm_fortest.pth.tar', device = '0', lstm_data_path = '../512w_fortest')

    att.train()
    att.extract_late(DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True))
    '''
    model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
    pretrained_dict = torch.load(args.pretrained_model)
    model_dict = model.state_dict()
    #model_dict.update(pretrained_dict)
    model_dict.update(pretrained_dict['state_dict'])
    model.load_state_dict(model_dict, strict=False)
    
    model.to(device)

    model._modules.get(hook_name).register_forward_hook(hook_feature)
    if args.loss_function != 'f':
        criterion = torch.nn.BCELoss().to(device)
    else:
        criterion = floss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    modelw = lstmnet()
    modelw.to(device)
    criterionw = nn.MSELoss().to(device)
    optimizerw = torch.optim.Adam(modelw.parameters(), lr=1e-4)

    if not args.train_lstm: # then load pretrained lstm
        trained_model = args.pretrained_lstm #'../savelstm/best_train3_lstmnet.pth.tar'
        pretrained_dict = torch.load(trained_model)
        model_dict = modelw.state_dict()
        model_dict.update(pretrained_dict)
        modelw.load_state_dict(model_dict)
        print('loaded pretrained lstm from ' + args.pretrained_lstm)
    else:
        if args.extract_lstm:
            extract_LSTM_training_data(save_path='../512w', trained_model='save/best_fusion.pth.tar', device='0', crop_size=3)
        from data.wdatas import wTrainData, wValData
        wTrainLoader = DataLoader(dataset=wTrainData, batch_size=1, shuffle=False, num_workers=0)
        wValLoader = DataLoader(dataset=wValData, batch_size=1, shuffle=False, num_workers=0)
    
        print('begin training lstm....')
        prev = 999
        prevt = 999
        loss_train = []
        loss_val = []
        for epoch in tqdm(range(args.num_epoch_lstm)):
            #lr = raw_input('please input lr:')
            #lr = float(lr)
            #adjust_learning_rate(optimizerw, epoch, lr)
            l = trainw(wTrainLoader, modelw, criterionw, optimizerw)
            loss_train.append(l)
            if l < prevt:
                torch.save(modelw.state_dict(), os.path.join(args.save_path, args.save_lstm))
            l = testw(wValLoader, modelw, criterionw)
            loss_val.append(l)
            if l<prev:
                prev=l
                torch.save(modelw.state_dict(), os.path.join(args.save_path, 'val'+args.save_lstm))
            plot_loss(loss_train, loss_val, os.path.join(args.save_path, args.lstm_save_img))
        print('lstm training finished!')
	
    model_late = late_fusion()
    model_late.to(device)
    if not args.train_late:  # then load pretrained late fusion model
        trained_model = args.pretrained_late
        pretrained_dict = torch.load(trained_model)
        model_dict = model_late.state_dict()
        model_dict.update(pretrained_dict)
        model_late.load_state_dict(model_dict)

    optimizer_late = torch.optim.Adam(model_late.parameters(), lr=1e-4)
    
    #vis_features(STTrainLoader, model, modelw, 'savelstm/3layerall/vistrainrelu/')

    if args.extract_late:
        extract_late(DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True), model, modelw)
        extract_late(DataLoader(dataset=STTrainData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True), model, modelw)

    del model, optimizer
    del modelw, optimizerw
    from data.lateDataset import lateDatasetTrain, lateDatasetVal
    train_loader = DataLoader(dataset = lateDatasetTrain, batch_size = args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset = lateDatasetVal, batch_size = args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    trainprev = 999
    valprev = 999
    loss_train = []
    loss_val = []
    for epoch in range(args.num_epoch):
        if args.train_late:
            print ('begin training model epoch %03d....'%epoch)
            loss, auc, aae = train_late(epoch, train_loader, model_late, criterion, optimizer_late)
            loss_train.append(loss)
            print('training, auc is %5f, aae is %5f'%(auc, aae))
            if loss < trainprev:
                torch.save({'state_dict': model_late.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, os.path.join(args.save_path, args.save_late))
                trainprev = loss
        print('begin validation...')
        loss, auc, aae = val_late(epoch, val_loader, model_late, criterion)
        loss_val.append(loss)
        print('val, auc is %5f, aae is %5f'%(auc, aae))
        plot_loss(loss_train, loss_val, os.path.join(args.save_path, args.late_save_img))
        if loss < valprev:
            torch.save({'state_dict': model_late.state_dict(), 'loss': loss, 'auc': auc, 'aae': aae}, os.path.join(args.save_path, 'val'+args.save_late))
            valprev = loss
        if not args.train_late:
            break
    '''
    