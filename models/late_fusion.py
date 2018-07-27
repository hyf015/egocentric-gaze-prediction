import torch
from torch.autograd import Variable
import torch.nn as nn
import math

class late_fusion(nn.Module):
    def __init__(self,):
        super(late_fusion, self).__init__()
        self.upsample = nn.Upsample(scale_factor=16)
        self.fusion = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 8, kernel_size=3, padding = 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 1, kernel_size=1, padding = 0)
                                    )
        self.final = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, f, g):
        #f = self.upsample(f)
        fused = torch.cat((f,g), dim = 1)
        fused = self.fusion(fused)
        fused = self.final(fused)
        return fused

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


if __name__ == '__main__':
    model = late_fusion()
    tensor1 = Variable(torch.randn(10,1,14,14))
    tensor2 = Variable(torch.randn(10,1,224,224))
    output = model(tensor1, tensor2)
    print(output.size())