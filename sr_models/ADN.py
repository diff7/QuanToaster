import torch
import torch.nn as nn

class AdaptiveNormalization(nn.Module):
    def __init__(self, filters, skip_mode=False):
        super().__init__()

        self.bn = nn.BatchNorm2d(filters)
        self.phi = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.phi.weight.data.fill_(1.5)
        self.phi.bias.data.fill_(0)
        self.skip_mode = skip_mode

    def forward(self, x, func, skip):
        
        s = torch.std(skip, dim=[1,2,3], keepdim=True)
        self.s = self.phi(s)
        
        if self.skip_mode:
            x_nm = func(x)
            out = x_nm + skip

        else:
            x_nm = self.bn(x)
            x_nm = func(x_nm)
            out = x_nm*self.s + skip

        return out