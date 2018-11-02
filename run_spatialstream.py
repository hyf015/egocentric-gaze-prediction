from utils import *
import cv2
import numpy as np
import torch
import argparse
from models.late_fusion import late_fusion
from scipy import ndimage

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', default='models/spatial.pth.tar', required=False,)
parser.add_argument('--trained_late', default='models/late.pth.tar', required=False,)
parser.add_argument('--dir', required=True)
parser.add_argument('--device', default='0', help='now only support single GPU')
args = parser.parse_args()
device = torch.device('cuda:'+args.device) if torch.cuda.is_available else 'cpu'

class VGG(nn.Module):
    
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        for param in self.features.parameters():
            param.requires_grad = False
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1),nn.ReLU(inplace=True),
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
        self._initialize_weights()

    def forward(self, x):
        xo = self.features(x)
        x = self.decoder(xo)
        y = self.final(x)
        return y, xo
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


model = VGG(make_layers(cfg['D'], 3))
trained_model = args.trained_model
pretrained_dict = torch.load(trained_model, map_location=device)
pretrained_dict = pretrained_dict['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()
model.to(device)
lf = late_fusion()
pretrained_dict = torch.load(args.trained_late, map_location=device)
pretrained_dict = pretrained_dict['state_dict']
lf.load_state_dict(pretrained_dict)
lf.eval()
lf.to(device)

def crop_feature1(feature, maxind, size):
    #maxind is gaze point
    H = feature.size(2)
    W = feature.size(3)
    fmax = np.array(maxind)
    fmax = fmax // 16  #downsize from 224 to 14
    fmax = np.clip(fmax, size//2, H-int(math.ceil(size/2.0)))
    print(fmax)
    cfeature = feature[:,:,int(fmax[0]-size//2):int(fmax[0]+int(math.ceil(size/2.0))),int(fmax[1]-size//2):int(fmax[1]+int(math.ceil(size/2.0)))]
    return cfeature

def get_weighted(chn_weight, feature):
    #chn_weight (512), feature(1,512,14,14)
    chn_weight = chn_weight.view(1,512,1,1)
    feature = feature * chn_weight
    feature = torch.sum(feature, 1)
    feature = feature - torch.min(feature)
    feature = feature / torch.max(feature)
    #feature = feature - torch.mean(feature)
    return feature.unsqueeze(0)

def totensor(im):
    im = im.transpose((2,0,1))
    im = torch.from_numpy(im)
    im = im.float().div(255)
    im = im.sub_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1)).div_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
    return im.unsqueeze(0)

def toim(ten):
    ten = ten.squeeze().cpu()
    #ten = ten.mul_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1)).add_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1))
    ten = ten.numpy()
    ten = (ten*255).astype(np.uint8)
    #ten = ten.transpose((2,0,1))
    return ten

ims = os.listdir(args.dir)
ims = [k for k in ims if 'img' in k]
for imname in ims:
    im = cv2.imread(os.path.join(args.dir, imname))
    im = cv2.resize(im, (224,224))
    im = totensor(im)
    im = im.to(device)
    with torch.no_grad():
        out, feat = model(im)
    im = toim(out)
    predicted = ndimage.measurements.center_of_mass(im)
    vec = crop_feature1(feat, predicted,3)
    vec = vec.contiguous().view(vec.size(0), vec.size(1), -1)
    vec = torch.mean(vec, 2).squeeze()
    weighted = get_weighted(vec, feat)
    weighted = torch.nn.functional.upsample(weighted, scale_factor=16, mode='bilinear')
    with torch.no_grad():
        fin = lf(out, weighted)
    fin = toim(fin)

    im = cv2.imread(os.path.join(args.dir, imname))
    colormap = cv2.applyColorMap(cv2.resize(fin, (im.shape[1], im.shape[0])), cv2.COLORMAP_JET)
    res = im * 0.7 + colormap * 0.3
    cv2.imwrite(os.path.join(args.dir, 'out_'+imname[3:]), res)
    print('result saved to '+ os.path.join(args.dir, 'out_'+imname[3:]))
