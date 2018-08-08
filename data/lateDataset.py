import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import math
from tqdm import tqdm

gtPath = '../gtea_gts'
listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
listGtFiles.sort()
listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
listValGtFiles.sort()
print('num of training samples: ', len(listGtFiles))


imgPath_s = '../new_pred'
listTrainFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' not in k]
#listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
listValFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' in k]
#listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
listTrainFiles.sort()
listValFiles.sort()
print('num of val samples: ', len(listValFiles))

featPath = '../new_feat'
listTrainFeats = [k for k in os.listdir(featPath) if 'Alireza' not in k]
listValFeats = [k for k in os.listdir(featPath) if 'Alireza' in k]
listTrainFeats.sort()
listValFeats.sort()
assert(len(listTrainFeats) == len(listTrainFiles))
assert(len(listValGtFiles) == len(listValFiles))

class lateDataset(Dataset):
    def __init__(self, imgPath_s, gtPath, featPath, listFiles, listGtFiles, listFeat):
        self.imgPath_s = imgPath_s
        self.gtPath = gtPath
        self.featPath = featPath
        self.listFeat = listFeat
        self.listFiles = listFiles
        self.listGtFiles = listGtFiles

    def __len__(self):
        return len(self.listGtFiles)

    def __getitem__(self, index):
        im = io.imread(self.imgPath_s + '/' + self.listFiles[index])
        gt = io.imread(self.gtPath + '/' + self.listGtFiles[index])
        feat = io.imread(self.featPath + '/' + self.listFeat[index])
        im = torch.from_numpy(im)
        gt = torch.from_numpy(gt)
        feat = torch.from_numpy(feat)
        im = im.float().div(255)
        gt = gt.float().div(255)
        feat = feat.float().div(255)
        im = im.unsqueeze(0)
        gt = gt.unsqueeze(0)
        feat = feat.unsqueeze(0)
        return {'im': im, 'gt': gt, 'feat': feat}


lateDatasetTrain = lateDataset(imgPath_s, gtPath, featPath, listTrainFiles, listGtFiles, listTrainFeats)
lateDatasetVal = lateDataset(imgPath_s, gtPath, featPath, listValFiles, listValGtFiles, listValFeats)

if __name__ == '__main__':
    a = DataLoader(dataset = lateDatasetTrain, batch_size = 10, shuffle=False, num_workers=1, pin_memory=True)
    print (len(a))
    for i in tqdm(a):
        pass
    a = DataLoader(dataset = lateDatasetVal, batch_size = 10, shuffle=False, num_workers=1, pin_memory=True)
    print (len(a))
    for i in tqdm(a):
        pass