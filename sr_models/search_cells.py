""" CNN cell for architecture search """
import torch.nn as nn
from sr_models import quant_ops as ops
from sr_models.ADN import AdaptiveNormalization as ADN


def summer(values, increments):
    return (v + i for v, i in zip(values, increments))


class Residual(nn.Module):
    def __init__(self, skip, body, skip_mode=True):
        super(Residual, self).__init__()
        self.skip = skip
        self.body = body
        self.adn = ADN(36, skip_mode=skip_mode)

    def forward(self, x, b_weights, s_weights):
        
        def func(x):
            return self.skip(x, s_weights) + self.body(x, b_weights)
        
        return self.adn(x, func, x)

    def fetch_info(self, b_weights, s_weights):
        flops = 0
        memory = 0
        for layer, weights in zip(
            (self.body, self.skip), (b_weights, s_weights)
        ):

            flops, memory = summer((flops, memory), layer.fetch_info(weights))
        return flops, memory


class CommonBlock(nn.Module):
    def __init__(
        self, c_fixed, c_init, bits, num_layers, gene_type="head", scale=4, aux_fp=True
    ):
        """
        Creates list of blocks of specific gene_type.
        """
        super(CommonBlock, self).__init__()

        self.net = nn.ModuleList()
        self.name = gene_type
        for i in range(num_layers):
            (
                c_in,
                c_out,
            ) = (c_fixed, c_fixed)

            if i == 0 and gene_type == "head":
                c_in = c_init
            elif gene_type == "tail":
                if (i + 1) == num_layers:
                    c_out = c_init
                if i == 0:
                    c_in = c_init
            elif gene_type == "upsample":
                c_in = c_fixed
                c_out = 3 * (scale ** 2)
            else:
                c_in = c_fixed
                c_out = c_fixed
            self.net.append(ops.MixedOp(c_in, c_out, bits, c_fixed, gene_type, aux_fp=aux_fp))

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
    def __init__(self, c_init, c_fixed, bits, scale, arch_pattern, body_cells, aux_fp=True, skip_mode=True):
        """
        SuperNet.

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
        self.skip_mode = skip_mode
        # Generate searchable network with shared weights
        self.head = CommonBlock(
            c_fixed, c_init, bits, arch_pattern["head"], gene_type="head", aux_fp=aux_fp
        )

        self.body = nn.ModuleList()
        for _ in range(body_cells):
            b = CommonBlock(
                c_fixed, c_init, bits, arch_pattern["body"], gene_type="body", aux_fp=aux_fp
            )
            s = CommonBlock(
                c_fixed, c_init, bits, arch_pattern["skip"], gene_type="skip", aux_fp=aux_fp
            )
            self.body.append(Residual(s, b, skip_mode=skip_mode))

        self.upsample = CommonBlock(
            c_fixed,
            c_init,
            bits,
            arch_pattern["upsample"],
            gene_type="upsample",
            aux_fp=aux_fp
        )
        self.pixel_up = nn.PixelShuffle(scale)

        self.tail = CommonBlock(
            c_fixed, c_init, bits, arch_pattern["tail"], gene_type="tail", aux_fp=aux_fp
        )

        self.adn_one = ADN(36, skip_mode=skip_mode)
        self.adn_two =  ADN(3, skip_mode=skip_mode)

    def forward(self, x, alphas):
        init = self.head(x, alphas["head"])
        x = init

        def func_body(x):
            for cell in self.body:
                x = cell(x, alphas["body"], alphas["skip"])
            return x

        x = self.adn_one(x, func_body, init)
        x = self.pixel_up(self.upsample(x, alphas["upsample"]))

        def func_tail(x):
            return self.tail(x, alphas["tail"])
        
        out = self.adn_two(x, func_tail, x)
        return  out

    def fetch_weighted_flops_and_memory(self, alphas):
        flops = 0
        memory = 0
        for func, name in [
            (self.head, "head"),
            (self.tail, "tail"),
            (self.upsample, "upsample"),
        ]:

            f, m = func.fetch_info(alphas[name])
            flops += f
            memory += m

        for cell in self.body:
            f, m = cell.fetch_info(alphas["body"], alphas["skip"])
            flops += f
            memory += m

        return flops, memory
