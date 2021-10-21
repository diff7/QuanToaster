""" CNN cell for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


class Residual(nn.Module):
    def __init__(self, skip, body):
        super().__init__()
        self.skip = skip
        self.body = body

    def forward(self, x, b_weights, s_weights):
        return (self.skip(x, s_weights) + self.body(x, b_weights)) * 0.2 + x

    def fetch_info(self, b_weights, s_weights):
        flops = 0
        memory = 0
        for layer, weights in zip(
            (self.body, self.skip), (b_weights, s_weights)
        ):

            flops, memory = summer((flops, memory), layer.fetch_info(weights))
        return flops, memory


class CommonBlock(nn.Module):
    def __init__(self, c_fixed, c_init, num_layers, gene_type="head", scale=4):
        super().__init__()

        self.net = nn.ModuleList()
        self.name = gene_type
        for i in range(num_layers):
            (
                c_in,
                c_out,
            ) = (c_fixed, c_fixed)

            if i == 0 and gene_type == "head":
                c_in = c_init
            elif i + 1 == num_layers and gene_type == "tail":
                c_out = c_init
            elif i == 0 and gene_type == "tail":
                c_in = c_init

            elif gene_type == "upsample":
                c_in = c_fixed
                c_out = 3 * (scale ** 2)
            else:
                c_in = c_fixed
                c_out = c_fixed

            self.net.append(ops.MixedOp(c_in, c_out, c_fixed, gene_type))

    def forward(self, x, alphas):
        for layer, a_w in zip(self.net, alphas):
            x = layer(x, a_w)
        return x

    def fetch_info(self, alphas):
        flops = 0
        memory = 0
        for layer, weight in zip(self.net, alphas):
            flops, memory = summer(
                (flops, memory), layer.fetch_weighted_info(weight)
            )
        return flops, memory


class SearchArch(nn.Module):
    def __init__(self, c_init, c_fixed, scale, arch_pattern, body_cells):
        """
        Args:
            body_cells: # of intermediate body blocks
            c_fixed: # of channels to work with
            c_init:  # of initial channels, usually 3
            scale: # downsampling scale

            arch_pattern : {'head':2, 'body':4, 'tail':3, 'skip'=1, 'upsample'=1}
        """

        super().__init__()
        self.body_cells = body_cells
        self.c_fixed = c_fixed  # 32, 64 etc
        self.c_init = c_init

        # Generate searchable network with shared weights
        self.head = CommonBlock(
            c_fixed, c_init, arch_pattern["head"], gene_type="head"
        )

        self.body = nn.ModuleList()
        for _ in range(body_cells):
            b = CommonBlock(
                c_fixed, c_init, arch_pattern["body"], gene_type="body"
            )
            s = CommonBlock(
                c_fixed, c_init, arch_pattern["skip"], gene_type="skip"
            )
            self.body.append(Residual(s, b))

        self.upsample = CommonBlock(
            c_fixed, c_init, arch_pattern["upsample"], gene_type="upsample"
        )
        self.pixel_up = nn.PixelShuffle(scale)

        self.tail = CommonBlock(
            c_fixed, c_init, arch_pattern["tail"], gene_type="tail"
        )

    def forward(self, x, alphas):
        init = self.head(x, alphas["head"])
        x = init
        for cell in self.body:
            x = cell(x, alphas["body"], alphas["skip"])
        x = self.pixel_up(self.upsample(x + init * 0.2, alphas["upsample"]))
        return self.tail(x, alphas["tail"]) * 0.2 + x

    def fetch_weighted_flops_and_memory(self, alphas):
        flops = 0
        memory = 0

        for func, name in [
            (self.head, "head"),
            (self.tail, "tail"),
            (self.upsample, "upsample"),
        ]:

            flops, memory = summer(
                (flops, memory), func.fetch_info(alphas[name])
            )

        for cell in self.body:
            flops, memory = summer(
                (flops, memory), cell.fetch_info(alphas["body"], alphas["skip"])
            )
        print(flops)
        print(alphas)
        return flops, memory