import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
from skimage import io
import math
from tqdm import tqdm
import cv2, os

from floss import floss
from data.STdatas import STTrainData, STValData
from utils import *
from models.LSTMnet import lstmnet
from models.SP import VGG_st_3dfuse
from extractLSTMw import extract_LSTM_training_data

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

def get_weighted(chn_weight, feature):
    #chn_weight (512), feature(1,512,14,14)
    chn_weight = chn_weight.view(1,512,1,1)
    feature = feature * chn_weight
    feature = torch.sum(feature, 1)
    feature = feature - torch.min(feature)
    feature = feature / torch.max(feature)
    #feature = feature - torch.mean(feature)
    return feature

class AT():
    def __init__(self, pretrained_model = None, pretrained_lstm = None, extract_lstm = False, \
            crop_size = 3, num_epoch_lstm = 30, lstm_save_img = 'loss_lstm.png',\
            save_path = 'save', save_name = 'best_lstm.pth.tar', device = '0', lstm_data_path = '../512w',):
        assert(pretrained_model is not None)
        self.device = torch.device('cuda:'+device)
        self.lstm = lstmnet().to(self.device)
        if pretrained_lstm is not None:
            pretrained_dict = torch.load(pretrained_lstm)
            model_dict = self.lstm.state_dict()
            model_dict.update(pretrained_dict)
            self.lstm.load_state_dict(model_dict)
            print('loaded pretrained lstm from ' + args.pretrained_lstm)
        self.criterion_lstm = nn.MSELoss().to(self.device)
        self.optimizer_lstm = torch.optim.Adam(self.lstm.parameters(), lr=1e-4)

        if extract_lstm:
            extract_LSTM_training_data(save_path=lstm_data_path, trained_model=pretrained_model, device=device, crop_size=crop_size)

        self.crop_size = crop_size
        self.num_epoch_lstm = num_epoch_lstm
        self.lstm_save_img = lstm_save_img
        self.save_path = save_path
        self.lstm_data_path = lstm_data_path
        self.save_name = save_name
        self.model = VGG_st_3dfuse(make_layers(cfg['D'], 3), make_layers(cfg['D'], 20))
        pretrained_dict = torch.load(pretrained_model)
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict['state_dict'])
        self.model.load_state_dict(model_dict, strict=False)
        self.model._modules.get(hook_name).register_forward_hook(hook_feature)
        from data.wdatas import wTrainData, wValData
        self.lstmTrainLoader = DataLoader(dataset=wTrainData, batch_size=1, shuffle=False, num_workers=0)
        self.lstmValLoader = DataLoader(dataset=wValData, batch_size=1, shuffle=False, num_workers=0)

    def trainLSTM(self):
        losses = AverageMeter()
        self.lstm.train()
        hidden = None
        feature_fusion = torch.ones(batch_size,512,1,1).to(device)
        currname = None
        tanh = nn.Tanh()
        relu = nn.ReLU()
        pred_chn_weight = None
        for i, sample in enumerate(self.lstmTrainLoader):
            #reset hidden state only when a video is over
            same = sample['same']
            if int(same) == 0:
                hidden = None

            inp = sample['input'].unsqueeze(0)   #(1, 1, 512)
            target = sample['gt'].unsqueeze(0)    #(1, 1, 512)

            if pred_chn_weight is not None:
                #pred_chn_weight = pred_chn_weight.squeeze()
                loss = self.criterion_lstm(pred_chn_weight, tanh(target))
                self.optimizer_lstm.zero_grad()
                loss.backward()
                self.optimizer_lstm.step()
                losses.update(loss.item())

            hidden = repackage_hidden(hidden)
            pred_chn_weight, hidden = self.lstm(inp, hidden)
            
        return losses.avg


    def testLSTM(self):
        losses = AverageMeter()
        self.lstm.eval()
        hidden = None
        feature_fusion = torch.ones(batch_size,512,1,1).to(device)
        currname = None
        tanh = nn.Tanh()
        relu = nn.ReLU()
        pred_chn_weight = None
        with torch.no_grad():
            for i, sample in enumerate(self.lstmValLoader):
                #reset hidden state only when a video is over
                same = sample['same']
                if int(same) == 0:
                    hidden = None

                inp = sample['input'].unsqueeze(0)   #(1, 1, 512)
                target = sample['gt'].unsqueeze(0)    #(1,1, 512)

                if pred_chn_weight is not None:
                    #pred_chn_weight = pred_chn_weight.squeeze()
                    loss = self.criterion_lstm(pred_chn_weight, tanh(target))
                    losses.update(loss.item())

                hidden = repackage_hidden(hidden)
                pred_chn_weight, hidden = self.lstm(inp, hidden)
            
        return losses.avg

    def train(self):
        print('begin training LSTM...')
        prev = 999
        prevt = 999
        loss_train = []
        loss_val = []
        for epoch in tqdm(range(self.num_epoch_lstm)):
            l = self.trainLSTM()
            loss_train.append(l)
            if l < prevt:
                torch.save(self.lstm.state_dict(), os.path.join(self.save_path, self.save_name))
            l = self.testLSTM()
            loss_val.append(l)
            if l<prev:
                prev=l
                torch.save(self.lstm.state_dict(), os.path.join(self.save_path, 'val'+self.save_name))
            plot_loss(loss_train, loss_val, os.path.join(self.save_path, self.lstm_save_img))
        print('lstm training finished!')

    def extract_late(self, st_loader, pred_folder = '../new_pred/', feat_folder = '../new_feat/'):
        # pred is the gaze prediction result of SP, feat is the output of AT.
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        if not os.path.exists(feat_folder):
            os.makedirs(feat_folder)
        global features_blobs
        model.eval()
        modelw.eval()
        hidden = None
        currname = None
        with torch.no_grad():
            for i, sample in tqdm(enumerate(st_loader)):
                currname = sample['imname'][0]
                fixsac = sample['fixsac']
                input_s = sample['image']
                target = sample['gt']
                input_t = sample['flow']
                input_s = input_s.float().to(device)
                input_t = input_t.float().to(device)
                target = target.float().to(device)
                input_var_s = input_s
                input_var_t = input_t
                target_var = target #(1,1,224,224)
                features_blobs = []
                output = self.model(input_var_s, input_var_t)  #(1,1,224,224)
                feature_s = features_blobs[0]  #(1,512,14,14)

                outim = output.cpu().data.numpy().squeeze() #(224,224)
                targetim = target_var.cpu().data.numpy().squeeze() #(224,224)
                outim = np.uint8(255*outim)
                cv2.imwrite(os.path.join(pred_folder,currname), outim)

                aae1, auc1, pred_gp = computeAAEAUC(outim,targetim)

                cfeature = crop_feature(feature_s, pred_gp, self.crop_size) #(1,512,3,3)
                cfeature = cfeature.contiguous()
                chn_weight = cfeature.view(cfeature.size(0), cfeature.size(1), -1)
                chn_weight = torch.mean(chn_weight, 2)  #(1,512)
                if int(fixsac) == 1:
                    feat = get_weighted(chn_weight, feature_s)
                else:
                    hidden = repackage_hidden(hidden)
                    chn_weight, hidden = self.lstm(chn_weight.unsqueeze(0), hidden)
                    chn_weight = chn_weight.squeeze(0)
                    feat = get_weighted(chn_weight, feature_s)
                feat = feat.cpu().data.numpy().squeeze()
                feat = np.uint8(255*feat)
                feat = cv2.resize(feat, (224,224))
                cv2.imwrite(os.path.join(feat_folder,currname), feat)