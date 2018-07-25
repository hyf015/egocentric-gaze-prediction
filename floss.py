import torch
import torch.nn as nn
import numpy as np

class floss(nn.Module):
    def __init__(self):
        super(floss,self).__init__()

    def forward(self, input, target):
        weights = self.build_weight_from_target(target)
        weights = torch.from_numpy(weights).cuda()
        loss = nn.functional.binary_cross_entropy(input, target, weight = weights)
        return loss

    def build_weight_from_target(self, target):
        target = target.data.cpu().numpy()
        batch_num = target.shape[0]
        image_width = target.shape[-1]
        weightmat = np.empty_like(target)
        weightmat.astype(float)

        for bb in range(batch_num):
            target_im = target[bb,:,:,:].squeeze()
            x, y = np.where(target_im == np.amax(target_im))
            x = x.mean()
            y = y.mean()
            #gp = np.array([x,y])
            a = np.arange(image_width)
            b = np.arange(image_width)
            a = a - x
            b = b - y
            a = np.tile(a, (image_width,1))
            b = np.tile(b, (image_width,1))
            a = np.transpose(a)
            dist = a**2 + b**2
            dist = (np.sqrt(dist) + 1) / image_width
            dist = np.reciprocal(dist)

            weightmat[bb,:,:,:] = dist

        return weightmat
