import torch
from torch.utils.data import Dataset, DataLoader
import os

trainPath = '512w/train'
testPath = '512w/test'
'''
listTrainFiles = [k for k in os.listdir(trainPath) if 'inp_' in k]
listTrainFiles.sort()
listTrainGts = [k for k in os.listdir(trainPath) if 'gt_' in k]
listTrainGts.sort()
listValFiles = [k for k in os.listdir(testPath) if 'inp_' in k]
listValFiles.sort()
listValGts = [k for k in os.listdir(testPath) if 'gt_' in k]
listValGts.sort()
'''
listTrainFiles = sorted(os.listdir(trainPath))
listValFiles = sorted(os.listdir(testPath))

print 'num of training samples: ', len(listTrainFiles)
print 'num of val samples: ', len(listValFiles)

class wDatasetTrain(Dataset):
    def __init__(self, trainPath, listTrainFiles):
        #imgPath is flow path, containing several subfolders
        self.trainPath = trainPath
        self.listTrainFiles = listTrainFiles
    
    def __len__(self):
        return len(self.listTrainFiles) - 1

    def __getitem__(self, index):
        inp = torch.load(os.path.join(self.trainPath , self.listTrainFiles[index]))
        gt = torch.load(os.path.join(self.trainPath , self.listTrainFiles[index+1]))
        same = listTrainFiles[index+1][:-14] == listTrainFiles[index][:-14]
        return {'input': inp, 'gt': gt, 'same': same}

class wDatasetVal(Dataset):
    def __init__(self, testPath, listValFiles):
        self.valPath = testPath
        self.listValFiles = listValFiles

    def __len__(self):
        return len(self.listValFiles) - 1

    def __getitem__(self, index):
        inp = torch.load(os.path.join(self.valPath , self.listValFiles[index]))
        gt = torch.load(os.path.join(self.valPath , self.listValFiles[index+1]))
        same = listValFiles[index+1][:-14] == listValFiles[index][:-14]
        return {'input': inp, 'gt': gt, 'same': same}


wTrainData = wDatasetTrain(trainPath, listTrainFiles)
wValData = wDatasetVal(testPath, listValFiles)

if __name__ == '__main__':
    wTrainLoader = DataLoader(dataset=wTrainData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    wValLoader = DataLoader(dataset=wValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print len(wTrainData), len(wValData)