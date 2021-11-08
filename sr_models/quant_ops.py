import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models.quant_conv import BaseConv
from sr_models.quant_conv_lsq import count_upsample_flops

import genotypes as gt


OPS = {
    "zero": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: Zero(
        stride, zero=0
    ),
    "skip_connect": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: Identity(
        shared
    ),
    "conv_5x1_1x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: FacConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        affine=affine,
        shared=shared,
    ),
    "conv_3x1_1x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: FacConv(
        C_in, C_out, bits, C_fixed, 3, stride, 1, affine=affine, shared=shared
    ),
    "simple_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in, C_out, bits, C_fixed, 3, stride, 1, affine=affine, shared=shared
    ),
    "simple_9x9": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in, C_out, bits, C_fixed, 9, stride, 4, affine=affine, shared=shared
    ),
    "simple_1x1": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in, C_out, bits, C_fixed, 1, stride, 0, affine=affine, shared=shared
    ),
    "simple_5x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in, C_out, bits, C_fixed, 5, stride, 2, affine=affine, shared=shared
    ),
    "simple_1x1_grouped_full": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        0,
        groups=C_in,
        affine=affine,
        shared=shared,
    ),
    "simple_3x3_grouped_full": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=C_in,
        affine=affine,
        shared=shared,
    ),
    "simple_5x5_grouped_full": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=C_in,
        affine=affine,
        shared=shared,
    ),
    "simple_1x1_grouped_3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        1,
        stride,
        0,
        groups=3,
        affine=affine,
        shared=shared,
    ),
    "simple_3x3_grouped_3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=3,
        affine=affine,
        shared=shared,
    ),
    "simple_5x5_grouped_3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: SimpleConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=3,
        affine=affine,
        shared=shared,
    ),
    "DWS_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DWS(
        C_in, C_out, bits, C_fixed, 3, stride, 1, affine=affine, shared=shared
    ),
    "DWS_5x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DWS(
        C_in, C_out, bits, C_fixed, 5, stride, 2, affine=affine, shared=shared
    ),
    "growth2_3x3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=1,
        affine=affine,
        shared=shared,
        growth=2,
    ),
    "growth2_5x5": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: GrowthConv(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=1,
        affine=affine,
        shared=shared,
        growth=2,
    ),
    "decenc_3x3_4": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=1,
        reduce=4,
        affine=affine,
        shared=shared,
    ),
    "decenc_3x3_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=1,
        reduce=2,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_2": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=1,
        reduce=2,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_4": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=1,
        reduce=4,
        affine=affine,
        shared=shared,
    ),
    "decenc_3x3_8": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=1,
        reduce=8,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_8": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=1,
        reduce=8,
        affine=affine,
        shared=shared,
    ),
    "decenc_3x3_4_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=3,
        reduce=4,
        affine=affine,
        shared=shared,
    ),
    "decenc_3x3_2_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=3,
        reduce=2,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_2_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=3,
        reduce=2,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_4_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=3,
        reduce=4,
        affine=affine,
        shared=shared,
    ),
    "decenc_3x3_8_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        3,
        stride,
        1,
        groups=3,
        reduce=8,
        affine=affine,
        shared=shared,
    ),
    "decenc_5x5_8_g3": lambda C_in, C_out, bits, C_fixed, stride, affine, shared: DecEnc(
        C_in,
        C_out,
        bits,
        C_fixed,
        5,
        stride,
        2,
        groups=3,
        reduce=8,
        affine=affine,
        shared=shared,
    ),
}


class BSup(nn.Module):
    def __init__(self, mode, repeat_factor, residual=False):
        super(BSup, self).__init__()
        self.residual = residual
        self.repeat_factor = repeat_factor // 3
        self.mode = mode

        if mode == "nearest":
            align_corners = None
        else:
            align_corners = True
        self.upsample = nn.Upsample(
            scale_factor=self.repeat_factor ** (1 / 2),
            mode=mode,
            align_corners=align_corners,
        )
        self.space_to_depth = torch.nn.functional.pixel_unshuffle

        self.last_shape_out = ()

    def fetch_info(self):
        flops = 0
        mem = 0
        flops = count_upsample_flops(self.mode, self.last_shape_out)
        if self.residual:
            flops += torch.prod(torch.tensor(self.last_shape_out)[1:])

        return flops, mem

    def mean_by_c(self, image, repeat_factor):
        b, d, w, h = image.shape
        image = image.reshape([b, d // repeat_factor, repeat_factor, w, h])
        return image.mean(2)

    def forward(self, x):
        shape_in = x.shape
        # Input C*scale, W, H
        # x_mean = C, W, H
        x_mean = self.mean_by_c(x, self.repeat_factor)
        # upsample C, W*scale^1/2, H^scale^1/2
        x_upsample = self.upsample(x_mean)
        self.last_shape_out = x_upsample.shape
        # x_upscaled = C*scale, W, H
        x_de_pscaled = self.space_to_depth(
            x_upsample, int(self.repeat_factor ** (1 / 2))
        )
        if self.residual:
            x = (x + x_de_pscaled) / 2
        else:
            x = x_de_pscaled

        assert (
            x.shape == shape_in
        ), f"shape mismatch in BSup {shape_in} {x.shape}"

        return x


class SimpleConv(BaseConv):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=True,
        shared=False,
    ):
        super().__init__(shared=shared)
        self.net = nn.Sequential(
            self.conv_func(
                in_channels=C_in,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            )
        )

        nn.BatchNorm2d(C_out, affine=False),

    def forward(self, x):
        return self.net(x)


class GrowthConv(BaseConv):
    """Standard conv -C_in -> C_in*growth -> C_in"""

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=False,
        growth=2,
        shared=False,
    ):
        super().__init__(shared=shared)
        self.net = nn.Sequential(
            self.conv_func(
                in_channels=C_in,
                out_channels=C_fixed * growth,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            # nn.ReLU(),
            self.conv_func(
                in_channels=C_fixed * growth,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            # nn.ReLU(),
            # nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.net(x)


class DecEnc(BaseConv):
    """Standard conv -C_in -> C_in*growth -> C_in"""

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=False,
        reduce=4,
        shared=False,
    ):
        super().__init__(shared=shared)
        self.net = nn.Sequential(
            self.conv_func(
                in_channels=C_in,
                out_channels=C_fixed // reduce,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            self.conv_func(
                in_channels=C_in // reduce,
                out_channels=C_fixed // reduce,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
            self.conv_func(
                in_channels=C_fixed // reduce,
                out_channels=C_out,
                bits=bits,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=affine,
            ),
        )

    def forward(self, x):
        return self.net(x)


class DWS(BaseConv):
    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_size,
        stride,
        padding,
        affine=True,
        growth=1,
        shared=False,
    ):
        super().__init__(shared=shared)

        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_out,  affine=affine, shared=shared),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_in * 4,
                bits=bits,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            self.conv_func(
                in_channels=C_in * 4,
                out_channels=C_in,
                bits=bits,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=C_in,
                bias=False,
            ),
            # nn.BatchNorm2d(C_out, affine=True),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FacConv(BaseConv):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(
        self,
        C_in,
        C_out,
        bits,
        C_fixed,
        kernel_length,
        stride,
        padding,
        affine=True,
        growth=1,
        shared=False,
    ):
        super().__init__(shared=shared)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_out,  affine=affine, shared=shared),
            self.conv_func(
                in_channels=C_in,
                out_channels=C_fixed * growth,
                bits=bits,
                kernel_size=(kernel_length, 1),
                stride=stride,
                padding=(padding, 0),
                dilation=1,
                bias=False,
            ),
            self.conv_func(
                in_channels=C_fixed * growth,
                out_channels=C_out,
                bits=bits,
                kernel_size=(1, kernel_length),
                stride=stride,
                padding=(0, padding),
                dilation=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.net(x)


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class Identity(BaseConv):
    def __init__(self, shared):
        super().__init__(shared=shared)

    def forward(self, x):
        return x


class Zero(BaseConv):
    def __init__(self, stride, zero=1e-15):
        super().__init__()
        self.stride = stride
        self.zero = zero

    def forward(self, x):
        if self.stride == 1:
            return x * self.zero

        # re-sizing by stride
        return x[:, :, :: self.stride, :: self.stride] * self.zero


class MixedOp(nn.Module):
    """Mixed operation"""

    def __init__(self, C_in, C_out, bits, C_fixed, gene_type, stride=1):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES_SR[gene_type]:
            func = OPS[primitive](
                C_in, C_out, bits, C_fixed, stride, affine=False, shared=True
            )
            self._ops.append(func)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """

        outs = []
        for alphas, op in zip(weights.chunk(len(self._ops)), self._ops):
            op.set_alphas(alphas)
            outs.append(op(x))
        return sum(outs)

    def fetch_weighted_info(self, weights):
        flops = 0
        memory = 0
        for alphas, op in zip(weights.chunk(len(self._ops)), self._ops):
            op.set_alphas(alphas)
            f, m = op.fetch_info()
            flops += f
            memory += m

        return flops, memory


if __name__ == "__main__":
    random_image = torch.randn(3, 36, 256, 256)

    C = 36
    # Keep stride 1 for all but GrowCo
    stride = 1

    PRIMITIVES_SR = [
        "skip_connect",  # identity
        "conv_5x1_1x5",
        "conv_3x1_1x3",
        "simple_3x3",
        "simple_1x1",
        "simple_5x5",
        "simple_1x1_grouped_full",
        "simple_3x3_grouped_full",
        "simple_5x5_grouped_full",
        "simple_1x1_grouped_3",
        "simple_3x3_grouped_3",
        "simple_5x5_grouped_3",
        "DWS_3x3",
        "DWS_5x5",
        "growth2_5x5",
        "growth2_3x3",
        "decenc_3x3_4",
        "decenc_3x3_2",
        "decenc_5x5_2",
        "decenc_5x5_8",
        "decenc_3x3_8",
        "decenc_3x3_4_g3",
        "decenc_3x3_2_g3",
        "decenc_5x5_2_g3",
        "decenc_5x5_8_g3",
        "decenc_3x3_8_g3",
    ]

    names = []
    flops = []
    for i, primitive in enumerate(PRIMITIVES_SR):
        func = OPS[primitive](C, C, C, stride, affine=False)
        conv = func

        x = conv(random_image)
        flops.append(conv.fetch_info()[0])
        names.append(primitive)
        print(i + 1, primitive, f"FLOPS: {conv.fetch_info()[0]:.2e}")

    max_flops = max(flops)
    flops_normalized = [f / max_flops for f in flops]
    names_sorted = [
        n
        for f, n in sorted(
            zip(flops_normalized, names), key=lambda pair: pair[0]
        )
    ]
    flops_normalized = sorted(flops_normalized)
    print("\n## SORTED AND NORMALIZED ##\n")
    for n, f in zip(names_sorted, flops_normalized):
        print(n, "FLOPS:", round(f, 5))