import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import math
from tqdm import tqdm

def build_temporal_list(imgPath, gtPath, listFolders, listGtFiles):
    imgx = []
    imgy = []
    for gt in listGtFiles:
        folder = gt[:-17]
        assert(folder in listFolders)
        number = gt[-9:-4]  #is a string
        xstr = []
        ystr = []
        for m in range(10):
            xstr.append(imgPath + '/' + folder + '/' + 'flow_x_' + '%05d'%(int(number) - m) + '.jpg')
            ystr.append(imgPath + '/' + folder + '/' + 'flow_y_' + '%05d'%(int(number) - m) + '.jpg')
        imgx.append(xstr)
        imgy.append(ystr)
    return imgx, imgy

class STDataset(Dataset):
    def __init__(self, imgPath, imgPath_s, gtPath, listFolders, listTrainFiles, listGtFiles, listfixsacTrain, fixsacPath):
        #imgPath is flow path, containing several subfolders
        self.listFolders = listFolders
        self.listGtFiles = listGtFiles
        self.imgPath = imgPath
        self.imgPath_s = imgPath_s
        self.listTrainFiles = listTrainFiles
        self.gtPath = gtPath
        self.imgx, self.imgy = build_temporal_list(imgPath, gtPath, self.listFolders, listGtFiles)
        self.fixsac = 'i'
        for file in listfixsacTrain:
            a=np.loadtxt(os.path.join(fixsacPath,file))
            ker = np.array([1,1,1])
            a = np.convolve(a, ker)
            a = a[1:-1]
            a = (a>0).astype(float)
            if type(self.fixsac)==type('i'):
                self.fixsac = a
            else:
                self.fixsac = np.concatenate((self.fixsac,a))
    
    def __len__(self):
        return len(self.listGtFiles)

    def __getitem__(self, index):
        im = cv2.imread(self.imgPath_s + '/' + self.listTrainFiles[index])
        im = im.transpose((2,0,1))
        im = torch.from_numpy(im)
        im = im.float().div(255)
        im = im.sub_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1)).div_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
        flowx = self.imgx[index]
        flowy = self.imgy[index]
        flowarr = []
        for flowi in range(10):
            currflowx = cv2.imread(flowx[flowi], 0)
            currflowy = cv2.imread(flowy[flowi], 0)
            flowarr.append(currflowx)
            flowarr.append(currflowy)
        gt = cv2.imread(self.gtPath + '/' + self.listGtFiles[index], 0)
        flowarr = np.stack(flowarr, axis=0)
        flowarr = torch.from_numpy(flowarr)
        flowarr = flowarr.div_(255.0)
        flowarr = flowarr.sub_(0.5)
        flowarr = flowarr.div(0.5)
        gt = torch.from_numpy(gt)
        gt = gt.float().div(255)
        gt = gt.unsqueeze(0)
        sample = {'image': im, 'flow': flowarr, 'gt': gt, 'fixsac': torch.FloatTensor([self.fixsac[index]]), 'imname': self.listTrainFiles[index]}
        return sample


if __name__ == '__main__':
    imgPath = '../gtea_imgflow'
    gtPath = '../gtea_gts'
    fixsacPath = '../fixsac'
    listFolders = [k for k in os.listdir(imgPath)]
    listFolders.sort()
    listGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' not in k]
    listGtFiles.sort()
    listValGtFiles = [k for k in os.listdir(gtPath) if 'Alireza' in k]
    listValGtFiles.sort()
    print('num of training samples: ', len(listGtFiles))

    listfixsacTrain = [k for k in os.listdir(fixsacPath) if 'Alireza' not in k]
    listfixsacVal = [k for k in os.listdir(fixsacPath) if 'Alireza' in k]
    listfixsacVal.sort()
    listfixsacTrain.sort()

    imgPath_s = '../gtea_images'
    listTrainFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' not in k]
    listValFiles = [k for k in os.listdir(imgPath_s) if 'Alireza' in k]

    listTrainFiles.sort()
    listValFiles.sort()
    print('num of val samples: ', len(listValFiles))
    STTrainData = STDataset(imgPath, imgPath_s, gtPath, listFolders, listTrainFiles, listGtFiles, listfixsacTrain, fixsacPath)
    STTrainLoader = DataLoader(dataset=STTrainData, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)

    STValData = STDataset(imgPath, imgPath_s, gtPath, listFolders, listValFiles, listValGtFiles, listfixsacVal, fixsacPath)
    STValLoader = DataLoader(dataset=STValData, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    print(len(STValLoader))
    print(len(STTrainLoader))
    for i in tqdm(STTrainLoader):
        pass
