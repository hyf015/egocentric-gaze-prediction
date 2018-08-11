import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
from skimage import io
import math
from tqdm import tqdm
import cv2, os

from utils import *
from data.STdatas import STTrainData, STValData
from models.SP import VGG_st_3dfuse
from models.LSTMnet import lstmnet

hook_name = 'features_s'

global features_blobs
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output)

def crop_feature_var(feature, maxind, size):
    # used only in vis features
    H = feature.size(2)
    W = feature.size(3)
    for b in range(feature.size(0)):
        ind = maxind[b].item()
        fmax = np.unravel_index(ind, (H,W))
        fmax = np.clip(fmax, int(size/2), H-int(math.ceil(size/2.0)))
        cfeature = feature[b,:,int(fmax[0]-size/2):int(fmax[0]+int(math.ceil(size/2.0))),int(fmax[1]-size/2):(fmax[1]+int(math.ceil(size/2.0)))]
        cfeature = cfeature.unsqueeze(0)
        if b==0:
            res = cfeature
        else:
            res = torch.cat((res, cfeature),0)
    return res


def vis_features(st_loader, model, modelw, savefolder):
    global features_blobs
    model.eval()
    modelw.eval()
    feature_fusion = torch.ones(batch_size,512,1,1).to(device)
    pred_chn_weight = None
    downsample = nn.AvgPool2d(16)
    hidden = None
    for i, sample in enumerate(st_loader):
        if i<100:
            continue
        if i>1000:
            return
        currname = sample['imname'][-1][:-4]
        input_s = sample['image']
        target = sample['gt']
        input_t = sample['flow']
        input_s = input_s.float().to(device)
        input_t = input_t.float().to(device)
        target = target.float().to(device)
        input_var_s = input_s
        input_var_t = input_t
        target_var = target

        target_flat = downsample(target_var).view(target_var.size(0), target_var.size(1), -1)
        _, maxind = torch.max(target_flat, 2)
        feature_fusion = torch.ones(batch_size,512,1,1).to(device)
        features_blobs = []
        _ = model(input_var_s, input_var_t, feature_fusion, i)
        feature_fusion = features_blobs[0]

        cfeature = crop_feature_var(feature_fusion, maxind, 5)
        chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
        chn_weight = torch.mean(chn_weight, 2)

        weighted_feature = feature_fusion * chn_weight.view(batch_size, 512, 1, 1)
        weighted_feature = torch.sum(weighted_feature, 1)
        weighted_feature = weighted_feature[0,:,:].data.cpu().numpy()
        weighted_feature = weighted_feature - np.amin(weighted_feature)
        weighted_feature = weighted_feature / np.amax(weighted_feature)
        weighted_feature = np.uint8(255*weighted_feature)
        weighted_feature = cv2.resize(weighted_feature, (224,224))
        img = cv2.imread('../gtea_images/' + sample['imname'][0])
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
        img = cv2.imread('../gtea_images/' + sample['imname'][0])
        heatmap = cv2.applyColorMap(weighted_feature, cv2.COLORMAP_JET)
        result = heatmap*0.3 + img*0.5
        cv2.imwrite(savefolder + 'noweight_' + sample['imname'][0], result)

        if pred_chn_weight is not None:
            weighted_feature = feature_fusion * pred_chn_weight.view(batch_size, 512, 1, 1)
            weighted_feature = torch.sum(weighted_feature, 1)
            weighted_feature = weighted_feature[0,:,:].data.cpu().numpy()
            weighted_feature = weighted_feature - np.amin(weighted_feature)
            weighted_feature = weighted_feature / np.amax(weighted_feature)
            weighted_feature = np.uint8(255*weighted_feature)
            weighted_feature = cv2.resize(weighted_feature, (224,224))

            img = cv2.imread('../gtea_images/' + sample['imname'][0])
            heatmap = cv2.applyColorMap(weighted_feature, cv2.COLORMAP_JET)
            result = heatmap*0.3 + img*0.5
            cv2.imwrite(savefolder + 'pred_' + sample['imname'][0], result)
        gaze = target_var.data.cpu().numpy()
        gaze = gaze[0,:,:,:].squeeze()
        io.imsave(savefolder + 'gaze_' + sample['imname'][0], gaze)


        #print chn_weight.size()   #(batch_size,512)
        chn_weight = chn_weight.unsqueeze(1)  #(seq_len, batch, input_size)
        chn_weight = chn_weight.to(device)
        hidden = repackage_hidden(hidden)
        pred_chn_weight, hidden = modelw(chn_weight, hidden)  #pred size (seq_len, batch, output_size) ie (batch_size, 1, 512)
        pred_chn_weight = pred_chn_weight.squeeze()
        feature_fusion = (pred_chn_weight+1)/2  #turn to range(0,1)
        feature_fusion = feature_fusion.view(batch_size, 512, 1, 1)

if __name__ == '__main__':
    STTrainLoader = DataLoader(dataset=STTrainData, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    STValLoader = DataLoader(dataset=STValData, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    device = torch.device('cuda:0')
    model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
    model_dict = model.state_dict()
    pretrained_dict = torch.load('save/best_fusion.pth.tar')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model._modules.get(hook_name).register_forward_hook(hook_feature)
    lstm = lstmnet()
    model_dict = lstm.state_dict()
    pretrained_dict = torch.load('save/valbest_lstm.pth.tar')
    model_dict.update(pretrained_dict)
    lstm.load_state_dict(model_dict)
    lstm.to(device)
    vis_features(STValLoader, model, lstm, 'vis')


