import torch, math
import torch.nn as nn

class VGG_st_3dfuse(nn.Module):
    def __init__(self, features_s, features_t):
        super(VGG_st_3dfuse, self).__init__()
        self.features_t = features_t
        self.features_s = features_s
        self.relu = nn.ReLU()
        self.fusion = nn.Conv3d(512, 512, kernel_size=(1,3,3), padding=(0,1,1))
        self.pool3d = nn.MaxPool3d(kernel_size=(2,1,1), padding=0)
        self.bn = nn.BatchNorm2d(512)
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding = 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding = 1),
                                        nn.ReLU(inplace=True),
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

    def forward(self, x_s, x_t):
        x_s = self.features_s(x_s)
        x_t = self.features_t(x_t)
        x_s = x_s.unsqueeze(2)
        x_t = x_t.unsqueeze(2)
        x_fused = torch.cat((x_s, x_t), 2)
        x_fused = self.fusion(x_fused) #(batch_size, 512, 2, 14, 14)
        
        x_fused = self.pool3d(x_fused) #(batch_size, 512, 1, 14, 14)
        x_fused = x_fused.squeeze_(2) #(batch_size, 512, 14, 14)
        x_fused = self.bn(x_fused)

        x_fused = self.relu(x_fused)
        x_fused = self.decoder(x_fused)
        x = self.final(x_fused)
        return x

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
