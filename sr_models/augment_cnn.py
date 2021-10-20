""" CNN for network augmentation """
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops
import genotypes as gt


class Residual(nn.Module):
    def __init__(self, skip, body):
        super().__init__()
        self.skip = skip
        self.body = body

    def forward(self, x):
        return (self.skip(x) + self.body(x)) * 0.2 + x


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

        fb = []
        for _ in range(blocks):
            b = gt.to_dag_sr(self.c_fixed, genotype.body, gene_type="body")
            s = gt.to_dag_sr(self.c_fixed, genotype.skip, gene_type="skip")
            fb.append(Residual(s, b))

        self.body = nn.Sequential(*fb)

        upsample = gt.to_dag_sr(
            self.c_fixed, genotype.upsample, gene_type="upsample"
        )

        self.upsample = nn.Sequential(upsample, nn.PixelShuffle(scale))
        self.tail = gt.to_dag_sr(self.c_fixed, genotype.tail, gene_type="tail")

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsample(x) + self.tail(x)
        return x

    def drop_path_prob(self, p):
        """Set drop path probability"""
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p

    def assertion_in(self, size_in):
        assert int(size_in[1]) == int(
            self.c_fixed
        ), f"Input size {size_in}, does not match fixed channels {self.c_fixed}"

    def fetch_flops(self):
        return sum(
            sum(op.fetch_info()[0] for op in block) for block in self.dag
        )
