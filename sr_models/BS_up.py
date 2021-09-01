import torch
import torch.nn as nn
from flops import BaseConv
from flops import count_upsample_flops


# TODO
"""
Two types of blocks
1. Grouped
2. Increase num filters in the middle
3. Test /  Find different stride size
4. Add / Remove BN
"""

OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),  # 5x5
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),  # 9x9
    "conv_7x1_1x7": lambda C, stride, affine: FacConv(
        C, C, 7, stride, 3, affine=affine
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
    def __init__(self, mode, scale, residual=False):
        super(BSup, self).__init__()
        self.residual = residual
        self.scale = scale
        self.mode = mode

        if mode == "nearest":
            align_corners = None
        else:
            align_corners = True
        self.upsample = nn.Upsample(
            scale_factor=scale ** (1 / 2),
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

    def mean_by_c(self, image, scale):
        b, d, w, h = image.shape
        image = image.reshape([b, d // scale, scale, w, h])
        return image.mean(2)

    def forward(self, x):
        shape_in = x.shape
        # Input C*scale, W, H
        # x_mean = C, W, H
        x_mean = self.mean_by_c(x, self.scale)
        # upsample C, W*scale^1/2, H^scale^1/2

        x_upsample = self.upsample(x_mean)
        self.last_shape_out = x_upsample.shape
        # x_upscaled = C*scale, W, H
        x_upscaled = self.space_to_depth(x_upsample, int(self.scale ** (1 / 2)))
        if self.residual:
            x = (x + x_upscaled) / 2
        else:
            x = x_upscaled

        assert (
            x.shape == shape_in
        ), f"shape mismatch in BSup {shape_in} {x.shape}"

        return x_upscaled


# bs_up = BSup("nearest", scale=9, residual=False)

# bs_im = bs_up.forward(image_repeat)

# plt.imshow(bs_up.mean_by_c(bs_im, 9).transpose(1, -1)[0])

# plt.imshow(bs_im.transpose(1,-1)[0])k


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


class StdConv(BaseConv):
    """Standard conv
    ReLU - Conv - BN
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            self.conv_func(
                C_in, C_out, kernel_size, stride, padding, bias=False
            ),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.net(x)


class FacConv(BaseConv):
    """Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(
        self, C_in, C_out, kernel_length, stride, padding, affine=True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            self.conv_func(
                C_in,
                C_in,
                (kernel_length, 1),
                stride,
                padding,
                bias=False,
            ),
            self.conv_func(
                C_in,
                C_out,
                (1, kernel_length),
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.net(x)


class DilConv(BaseConv):
    """(Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine=True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            self.conv_func(C_in, C_out, 1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.net(x)


class SepConv(BaseConv):
    """Depthwise separable conv
    DilConv(dilation=1) * 2
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(
                C_in,
                C_in,
                kernel_size,
                stride,
                padding,
                dilation=1,
                affine=affine,
            ),
            DilConv(
                C_in,
                C_out,
                kernel_size,
                1,
                padding,
                dilation=1,
                affine=affine,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Identity(BaseConv):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(BaseConv):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0

        # re-sizing by stride
        return x[:, :, :: self.stride, :: self.stride] * 0.0


class FactorizedReduce(BaseConv):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.conv2 = self.conv_func(
            C_in, C_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


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
        ), f"Output size {size_in} does not match input size {size_out}, called from {self.func_name}"

    def forward(self, x):
        b, c, w, h = x.shape
        self.assertion_in((b, c, w, h))
        x = self.func(x)
        self.assertion_out((b, c, w, h), x.shape)
        return x

    def fetch_info(self):
        return self.func.fetch_info()


class MixedOp(nn.Module):
    """Mixed operation"""

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))

    def fetch_weighted_flops(self, weights):
        # s = 0
        # for w, op in zip(weights, self._ops):
        #     s += w * op.fetch_info()[0]
        #     print(type(op).__name__, w, op.fetch_info()[0])

        return sum(w * op.fetch_info()[0] for w, op in zip(weights, self._ops))

    def fetch_weighted_memory(self, weights):
        return sum(w * op.fetch_info()[1] for w, op in zip(weights, self._ops))


if __name__ == "__main__":
    random_image = torch.randn(3, 9, 10, 10)

    C = 9
    # Keep stride 1 for all but GrowCo
    stride = 1
    PRIMITIVES = [
        # "skip_connect",  # identity
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
        "bs_up_bicubic_residual",
        "bs_up_nearest_residual",
        "bs_up_bilinear_residual",
        "bs_up_bicubic",
        "bs_up_nearest",
        "bs_up_bilinear",
        "none",
    ]

    for primitive in PRIMITIVES:
        func = OPS[primitive](C, stride, affine=False)
        conv = AssertWrapper(func, channels=C)

        x = conv(random_image)
        print(primitive, "shape", x.shape, f"FLOPS: {conv.fetch_info()[0]}")
