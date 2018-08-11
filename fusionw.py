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


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    att = AT(pretrained_model =args.pretrained_model, pretrained_lstm = args.pretrained_lstm, extract_lstm = False, \
            crop_size = 3, num_epoch_lstm = 30, lstm_save_img = 'loss_lstm_fortest.png',\
            save_path = 'save', save_name = 'best_lstm_fortest.pth.tar', device = '0', lstm_data_path = '../512w_fortest')

    att.train()
    att.extract_late(DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True))
    lf = LF(pretrained_model = None, save_path = 'save', late_save_img = 'loss_late.png',\
            save_name = 'best_late.pth.tar', device = '0', late_pred_path = '../new_pred', num_epoch = 10,\
            late_feat_path = '../new_feat', gt_path = '../gtea_gts', val_name = 'Alireza', batch_size = 32,\
            loss_function = 'f')
    lf.train()