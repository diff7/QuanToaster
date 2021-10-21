""" CNN for network augmentation """
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops
import genotypes as gt


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


class Residual(nn.Module):
    def __init__(self, skip, body):
        super().__init__()
        self.skip = skip
        self.body = body

    def forward(self, x):
        return (self.skip(x) + self.body(x)) * 0.2 + x

    def fetch_weighted_info(self):
        flops = 0
        memory = 0
        for layer in (self.skip, self.body):
            flops, memory = summer((flops, memory), layer.fetch_info())
        return flops, memory


class AugmentCNN(nn.Module):
    """Augmented CNN model"""

    def __init__(self, c_in, c_fixed, scale, genotype, blocks=4):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.c_fixed = c_fixed  # c_init * repeat_factor
        self.repeat_factor = c_in * (scale ** 2)

        self.head = gt.to_dag_sr(self.c_fixed, genotype.head, gene_type="head")

        self.body = nn.ModuleList()
        for _ in range(blocks):
            b = gt.to_dag_sr(self.c_fixed, genotype.body, gene_type="body")
            s = gt.to_dag_sr(self.c_fixed, genotype.skip, gene_type="skip")
            self.body.append(Residual(s, b))

        upsample = gt.to_dag_sr(
            self.c_fixed, genotype.upsample, gene_type="upsample"
        )

        self.upsample = nn.Sequential(upsample, nn.PixelShuffle(scale))
        self.tail = gt.to_dag_sr(self.c_fixed, genotype.tail, gene_type="tail")

    def forward(self, x):
        init = self.head(x)
        x = init
        for cell in self.body:
            x = cell(x)
        x = self.upsample(x + init)
        return self.tail(x) * 0.2 + x

    def fetch_flops(self):
        flops = 0
        memory = 0

        for func in [self.head, self.body, self.tail, "tail", self.upsample]:
            flops, memory = summer((flops, memory), func.fetch_info())

        return flops, memory
