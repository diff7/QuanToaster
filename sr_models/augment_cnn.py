""" CNN for network augmentation """
import torch
import torch.nn as nn
import genotypes as gt
from sr_models.quant_conv_lsq import QAConv2d


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


 
class ADM(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.bn = nn.BatchNorm2d(filters)
        self.phi = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.phi.weight.data.fill_(1/4)
        self.phi.bias.data.fill_(0)

    def forward(self, x, func,skip):
        s = torch.std(skip, dim=[1,2,3], keepdim=True)
        self.s = self.phi(s)

        x_nm = self.bn(x)
        x_nm = func(x_nm)

        return x_nm*self.s + skip

class Residual(nn.Module):
    def __init__(self, skip, body, rf):
        super().__init__()
        self.skip = skip
        self.body = body
        self.rf = rf

        self.adaskip = ADM(36)

    def forward(self, x):
        def func(x):
            return (self.skip(x) + self.body(x))

        return self.adaskip(x, func, x)

    def fetch_weighted_info(self):
        flops = 0
        memory = 0
        for layer in (self.skip, self.body):
            flops, memory = summer((flops, memory), layer.fetch_info())
        return flops, memory


class AugmentCNN(nn.Module):
    """Searched CNN model for final training"""

    def __init__(self, c_in, c_fixed, scale, genotype, blocks=4, rf=1):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.rf = rf
        self.c_fixed = c_fixed
        self.head = gt.to_dag_sr(
            self.c_fixed, genotype.head, gene_type="head", c_in=c_in
        )

        self.body = nn.ModuleList()
        for _ in range(blocks):
            b = gt.to_dag_sr(
                self.c_fixed, genotype.body, gene_type="body", c_in=c_in
            )
            s = gt.to_dag_sr(
                self.c_fixed, genotype.skip, gene_type="skip", c_in=c_in
            )
            self.body.append(Residual(s, b, rf=self.rf))

        upsample = gt.to_dag_sr(
            self.c_fixed, genotype.upsample, gene_type="upsample"
        )

        self.upsample = nn.Sequential(upsample, nn.PixelShuffle(scale))
        self.tail = gt.to_dag_sr(
            self.c_fixed, genotype.tail, gene_type="tail", c_in=c_in
        )
        self.quant_mode = True

        self.adaskip_one = ADM(36)
        self.adaskip_two = ADM(3)

    def forward(self, x):

        self.stats = dict()
        self.stats['std'] = dict()
        self.stats['learnable'] = dict()
        self.stats['learnable']['mean'] = dict()
        self.stats['learnable']['std'] = dict()
        self.stats['learnable']['eps'] = dict()

        init = self.head(x)
        x = init
        self.stats['std']["head"] = torch.std(
            init, dim=[1, 2, 3], keepdim=True
        ).flatten()[0]

        def func(x):
            for cell in self.body:
                x = cell(x)
            return x

        x = self.upsample(self.adaskip_one(x, func,init))

        
        self.stats['learnable']['std']["body_out"] = torch.mean(self.adaskip_one.s)


        self.stats['std']["body"] = torch.std(
            x, dim=[1, 2, 3], keepdim=True
        ).flatten()[0]


        tail = self.adaskip_two(x, self.tail, x)

        self.stats['std']["tail"] = torch.std(
            tail, dim=[1, 2, 3], keepdim=True
        ).flatten()[0]

        self.stats['learnable']['std']["tail"] = torch.mean(self.adaskip_two.s)

        


        return tail

    def set_fp(self):
        if self.quant_mode == True:
            for m in self.modules():
                if isinstance(m, QAConv2d):
                    m.set_fp()
            self.quant_mode = False

    def set_quant(self):
        if self.quant_mode == False:
            for m in self.modules():
                if isinstance(m, QAConv2d):
                    m.set_quant()
            self.quant_mode = True

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        for m in self.modules():
            if isinstance(m, QAConv2d):
                b, m = m._fetch_info()
                sum_flops += b
                sum_memory = m
        return sum_flops, sum_memory
