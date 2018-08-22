import torch
from torch.utils.data import Dataset, DataLoader
import os

'''
trainPath = '../512w/train'
testPath = '../512w/test'
listTrainFiles = sorted(os.listdir(trainPath))
listValFiles = sorted(os.listdir(testPath))

print ('num of training samples: ', len(listTrainFiles))
print ('num of val samples: ', len(listValFiles))


class wDatasetTrain(Dataset):
    def __init__(self, trainPath='../512w/train', listTrainFiles=sorted(os.listdir('../512w/train'))):
        #imgPath is flow path, containing several subfolders
        self.trainPath = trainPath
        self.listTrainFiles = sorted(os.listdir(trainPath))
    
    def __len__(self):
        return len(self.listTrainFiles) - 1

    def __getitem__(self, index):
        inp = torch.load(os.path.join(self.trainPath , self.listTrainFiles[index]))
        gt = torch.load(os.path.join(self.trainPath , self.listTrainFiles[index+1]))
        same = listTrainFiles[index+1][:-14] == listTrainFiles[index][:-14]
        return {'input': inp, 'gt': gt, 'same': same}

class wDatasetVal(Dataset):
    def __init__(self, testPath='../512w/test', listValFiles=sorted(os.listdir('../512w/test'))):
        self.valPath = testPath
        self.listValFiles = sorted(os.listdir(testPath))

    def __len__(self):
        return len(self.listValFiles) - 1

    def __getitem__(self, index):
        inp = torch.load(os.path.join(self.valPath , self.listValFiles[index]))
        gt = torch.load(os.path.join(self.valPath , self.listValFiles[index+1]))
        same = listValFiles[index+1][:-14] == listValFiles[index][:-14]
        return {'input': inp, 'gt': gt, 'same': same}
'''
class lstmDataset(Dataset):
    def __init__(self, Path='../512w/test', name=None):
        self.Path = Path
        if name is None:
            self.listFiles = sorted(os.listdir(Path))
        else:
            self.listFiles = sorted([k for k in os.listdir(Path) if name in k])

    def __len__(self):
        return len(self.listFiles) - 1

    def __getitem__(self, index):
        inp = torch.load(os.path.join(self.Path , self.listFiles[index]))
        gt = torch.load(os.path.join(self.Path , self.listFiles[index+1]))
        same = self.listFiles[index+1][:-14] == self.listFiles[index][:-14]
        return {'input': inp, 'gt': gt, 'same': same}

if __name__ == '__main__':
    wTrainData =lstmDataset('../512w/train')
    wValData = lstmDataset('../512w/test')
    wTrainLoader = DataLoader(dataset=wTrainData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    wValLoader = DataLoader(dataset=wValData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print (len(wTrainData), len(wValData))