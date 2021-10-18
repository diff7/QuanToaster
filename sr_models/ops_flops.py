import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models.flops import BaseConv
from sr_models.flops import count_upsample_flops

import genotypes as gt


# check without skip no drop path
# check with SGD
# lower exp is sinlge path

OPS = {
    "none": lambda C, stride, affine: Zero(stride, zero=0),
    "skip_connect": lambda C, stride, affine: Identity(),
    "conv_5x1_1x5": lambda C, stride, affine: FacConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "conv_3x1_1x3": lambda C, stride, affine: FacConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "simple_3x3": lambda C, stride, affine: SimpleConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "simple_9x9": lambda C, stride, affine: SimpleConv(
        C, C, 9, stride, 4, affine=affine
    ),
    "simple_1x1": lambda C, stride, affine: SimpleConv(
        C, C, 1, stride, 0, affine=affine
    ),
    "simple_5x5": lambda C, stride, affine: SimpleConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "simple_1x1_grouped_full": lambda C, stride, affine: SimpleConv(
        C, C, 1, stride, 0, groups=C, affine=affine
    ),
    "simple_3x3_grouped_full": lambda C, stride, affine: SimpleConv(
        C, C, 3, stride, 1, groups=C, affine=affine
    ),
    "simple_5x5_grouped_full": lambda C, stride, affine: SimpleConv(
        C, C, 5, stride, 2, groups=C, affine=affine
    ),
    "simple_1x1_grouped_3": lambda C, stride, affine: SimpleConv(
        C, C, 1, stride, 0, groups=3, affine=affine
    ),
    "simple_3x3_grouped_3": lambda C, stride, affine: SimpleConv(
        C, C, 3, stride, 1, groups=3, affine=affine
    ),
    "simple_5x5_grouped_3": lambda C, stride, affine: SimpleConv(
        C, C, 5, stride, 2, groups=3, affine=affine
    ),
    "DWS_3x3": lambda C, stride, affine: DWS(C, C, 3, stride, 1, affine=affine),
    "DWS_5x5": lambda C, stride, affine: DWS(C, C, 5, stride, 2, affine=affine),
    "growth2_3x3": lambda C, stride, affine: GrowthConv(
        C, C, 3, stride, 1, groups=1, affine=affine, growth=2
    ),
    "growth2_5x5": lambda C, stride, affine: GrowthConv(
        C, C, 5, stride, 2, groups=1, affine=affine, growth=2
    ),
    "decenc_3x3_4": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=1, reduce=4, affine=affine
    ),
    "decenc_3x3_2": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=1, reduce=2, affine=affine
    ),
    "decenc_5x5_2": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=1, reduce=2, affine=affine
    ),
    "decenc_5x5_4": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=1, reduce=4, affine=affine
    ),
    "decenc_3x3_8": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=1, reduce=8, affine=affine
    ),
    "decenc_5x5_8": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=1, reduce=8, affine=affine
    ),
    "decenc_3x3_4_g3": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=3, reduce=4, affine=affine
    ),
    "decenc_3x3_2_g3": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=3, reduce=2, affine=affine
    ),
    "decenc_5x5_2_g3": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=3, reduce=2, affine=affine
    ),
    "decenc_5x5_4_g3": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=3, reduce=4, affine=affine
    ),
    "decenc_3x3_8_g3": lambda C, stride, affine: DecEnc(
        C, C, 3, stride, 1, groups=3, reduce=8, affine=affine
    ),
    "decenc_5x5_8_g3": lambda C, stride, affine: DecEnc(
        C, C, 5, stride, 2, groups=3, reduce=8, affine=affine
    ),
    "bs_up_bicubic_residual": lambda C, stride, affine: BSup(
        "bicubic",
        C,
        residual=True,
    ),
    "bs_up_nearest_residual": lambda C, stride, affine: BSup(
        "nearest",
        C,
        residual=True,
    ),
    "bs_up_bilinear_residual": lambda C, stride, affine: BSup(
        "bilinear",
        C,
        residual=True,
    ),
    "bs_up_bicubic": lambda C, stride, affine: BSup(
        "bicubic",
        C,
        residual=False,
    ),
    "bs_up_nearest": lambda C, stride, affine: BSup(
        "nearest",
        C,
        residual=False,
    ),
    "bs_up_bilinear": lambda C, stride, affine: BSup(
        "bilinear",
        C,
        residual=False,
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
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            self.conv_func(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            # nn.BatchNorm2d(C_out, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GrowthConv(BaseConv):
    """Standard conv -C_in -> C_in*growth -> C_in"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=False,
        growth=2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            self.conv_func(
                C_in,
                C_in * growth,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            nn.GELU(),
            self.conv_func(
                C_in * growth,
                C_out,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            nn.GELU(),
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
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        affine=False,
        reduce=4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            self.conv_func(
                C_in,
                C_in // reduce,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            nn.GELU(),
            self.conv_func(
                C_in // reduce,
                C_in // reduce,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            nn.GELU(),
            self.conv_func(
                C_in // reduce,
                C_in,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=affine,
            ),
            nn.GELU(),
            # nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.net(x)


class DWS(BaseConv):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, affine=True, growth=1
    ):
        super().__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_out, affine=affine),
            self.conv_func(
                C_in,
                C_in * 4,
                1,
                1,
                0,
                bias=False,
            ),
            nn.GELU(),
            self.conv_func(
                C_in * 4, C_in, kernel_size, 1, padding, bias=False, groups=C_in
            ),
            # nn.BatchNorm2d(C_out, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FacConv(BaseConv):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(
        self, C_in, C_out, kernel_length, stride, padding, affine=True, growth=1
    ):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm2d(C_out, affine=affine),
            self.conv_func(
                C_in,
                C_in * growth,
                (kernel_length, 1),
                stride,
                (padding, 0),
                bias=False,
            ),
            nn.GELU(),
            self.conv_func(
                C_in * growth,
                C_out,
                (1, kernel_length),
                stride,
                (0, padding),
                bias=False,
            ),
            nn.GELU(),
            # nn.BatchNorm2d(C_out, affine=True),
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


class DropPath_(BaseConv):
    def __init__(self, p=0.0):
        """[!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return "p={}, inplace".format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class Identity(BaseConv):
    def __init__(self):
        super().__init__()

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


class AssertWrapper(nn.Module):
    """
    1. Checks that image size does not change.
    2. Checks that mage input channels are the same as output channels.
    """

    def __init__(self, func, channels):

        super(AssertWrapper, self).__init__()
        self.channels = channels
        self.func = func
        self.func_name = func.__class__.__name__

    def assertion_in(self, size_in):
        assert (
            size_in[1] == self.channels
        ), f"Input size {size_in}, does not match fixed channels {self.channels}, called from {self.func_name}"

    def assertion_out(self, size_in, size_out):
        assert (
            size_in == size_out
        ), f"Output size {size_out} does not match input size {size_in}, called from {self.func_name}"

    def forward(self, x):
        b, c, w, h = x.shape
        self.assertion_in((b, c, w, h))
        x = self.func(x) + x
        self.assertion_out((b, c, w, h), x.shape)
        return

    def fetch_info(self):
        return self.func.fetch_info()


class MixedOp(nn.Module):
    """Mixed operation"""

    def __init__(self, C, stride, first=False):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES_SR:
            # avoid zero connection at the first node
            if first and primitive == "zero":
                continue
            # print(primitive, "channels:", C)
            func = OPS[primitive](C, stride, affine=True)
            self._ops.append(AssertWrapper(func, channels=C))

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """

        return sum(w * op(x) for w, op in zip(weights, self._ops))

    def fetch_weighted_flops(self, weights):
        return sum(w * op.fetch_info()[0] for w, op in zip(weights, self._ops))

    def fetch_weighted_memory(self, weights):
        return sum(w * op.fetch_info()[1] for w, op in zip(weights, self._ops))


if __name__ == "__main__":
    random_image = torch.randn(3, 48, 256, 256)

    C = 48
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
        # "growth4_3x3",
        # "none",
    ]

    names = []
    flops = []
    for i, primitive in enumerate(PRIMITIVES_SR):
        func = OPS[primitive](C, stride, affine=True)
        conv = AssertWrapper(func, channels=C)

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
