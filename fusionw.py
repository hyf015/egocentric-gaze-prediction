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
parser.add_argument('--lr', type=float, default=1e-7, required=False, help='lr for LF Adam')
parser.add_argument('--late_save_img', default='loss_late.png', required=False)
parser.add_argument('--pretrained_model', default='save/best_fusion.pth.tar', required=False)
parser.add_argument('--pretrained_lstm', default=None, required=False)
parser.add_argument('--pretrained_late', default=None, required=False)
parser.add_argument('--lstm_save_img', default='loss_lstm.png', required=False)
parser.add_argument('--save_lstm', default='best_lstm.pth.tar', required=False)
parser.add_argument('--save_late', default='best_late.pth.tar', required=False)
parser.add_argument('--save_path', default='save', required=False)
parser.add_argument('--loss_function', default='f', required=False)
parser.add_argument('--num_epoch', type=int, default=10, required=False)
parser.add_argument('--num_epoch_lstm', type=int, default=30, required=False)
parser.add_argument('--extract_lstm', action='store_true')
parser.add_argument('--extract_lstm_path', default='../512w', required=False)
parser.add_argument('--train_lstm', action='store_true')
parser.add_argument('--train_late', action='store_true')
parser.add_argument('--extract_late', action='store_true')
parser.add_argument('--extract_late_pred_folder', default='../new_pred/', required=False)
parser.add_argument('--extract_late_feat_folder', default='../new_feat/', required=False)
parser.add_argument('--device', default='0')
parser.add_argument('--val_name', default='Alireza', required=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--crop_size', type=int, default=3)
args = parser.parse_args()

device = torch.device('cuda:'+args.device)

batch_size = args.batch_size


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    att = AT(pretrained_model =args.pretrained_model, pretrained_lstm = args.pretrained_lstm, extract_lstm = args.extract_lstm, \
            crop_size = args.crop_size, num_epoch_lstm = args.num_epoch_lstm, lstm_save_img = args.lstm_save_img,\
            save_path = args.save_path, save_name = args.save_lstm, device = args.device, lstm_data_path = args.extract_lstm_path)
    
    if args.train_lstm:
        att.train()
    
    if args.extract_late:
        att.extract_late(DataLoader(dataset=STValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True), args.extract_late_pred_folder, args.extract_late_feat_folder)
    
    lf = LF(pretrained_model = args.pretrained_late, save_path = args.save_path, late_save_img = args.late_save_img,\
            save_name = args.save_late, device = args.device, late_pred_path = args.extract_late_pred_folder, num_epoch = args.num_epoch,\
            late_feat_path = args.extract_late_feat_folder, gt_path = '../gtea_gts', val_name = args.val_name, batch_size = args.batch_size,\
            loss_function = args.loss_function, lr=args.lr)
    if args.train_late:
        lf.train()
    else:
        lf.val()