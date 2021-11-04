import torch
import torch.nn as nn
from sr_models import quant_ops as ops_sr
from sr_models.quant_conv import BaseConv


class FromGENE(nn.Module):

    """
    Genotype_SR(normal=[[('growth2_5x5', 0)],
                           [('sep_conv_5x5', 1), ('growth2_3x3', 0)],
                           [('sep_conv_3x3', 1), ('growth4_3x3', 2)],
                           [('growth4_3x3', 3), ('skip_connect', 0)]],
                           normal_concat=range(4, 6))

    """

    def __init__(self, c_init, repeat_factor):
        super().__init__()

        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor

        self.growth2_5x5 = ops_sr.OPS["growth2_5x5"](self.c_fixed, 1, True)
        self.growth2_3x3 = ops_sr.OPS["growth2_3x3"](self.c_fixed, 1, True)
        self.sep_conv_5x5 = ops_sr.OPS["sep_conv_5x5"](self.c_fixed, 1, True)
        self.sep_conv_3x3 = ops_sr.OPS["sep_conv_3x3"](self.c_fixed, 1, True)
        self.growth4_3x3 = ops_sr.OPS["growth4_3x3"](self.c_fixed, 1, True)
        self.growth4_3x3_final = ops_sr.OPS["growth4_3x3"](
            self.c_fixed, 1, True
        )

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
        )

    def forward(self, x0):
        x0 = torch.repeat_interleave(x0, self.repeat_factor, 1)
        x1 = self.growth2_5x5(x0)
        x2 = self.sep_conv_5x5(x1) + self.growth2_3x3(x1)

        x3 = self.growth4_3x3(x2) + self.sep_conv_3x3(x1)
        x4 = self.growth4_3x3_final(x3) + x1

        s_out = x4

        out = self.pixelup(s_out)
        x_residual = self.pixelup(x0)
        return out + x_residual


class SRESPCN(BaseConv):
    def __init__(self, scale_factor):
        super(SRESPCN, self).__init__()

        # Feature mapping
        self.espcn = nn.Sequential(
            self.conv_func(3, 64, kernel=5, stride=1, padding=2),
            nn.Tanh(),
            self.conv_func(64, 32, kernel=3, stride=1, padding=1),
            nn.Tanh(),
            self.conv_func(
                32, 3 * (scale_factor ** 2), kernel=3, stride=1, padding=1
            ),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid(),
        )

        self.srcnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.espcn(x)
        x = self.srcnn(x) + x
        return x


class ESPCN(BaseConv):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()

        # Feature mapping
        self.feature_maps = nn.Sequential(
            self.conv_func(3, 64, kernel=5, stride=1, padding=2),
            nn.Tanh(),
            self.conv_func(64, 32, kernel=3, stride=1, padding=1),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            self.conv_func(
                32, 3 * (scale_factor ** 2), kernel=3, stride=1, padding=1
            ),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        out = self.feature_maps(inputs)
        out = self.sub_pixel(out)
        return out


class ResidualBlock(nn.Module):
    """Base Residual Block"""

    def __init__(self, channels: int, kernel_size: int, activation):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class UpsampleBlock(nn.Module):
    """Base PixelShuffle Upsample Block"""

    def __init__(
        self, n_upsamples: int, channels: int, kernel_size: int, activation
    ):
        super().__init__()

        layers = []
        for _ in range(n_upsamples):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels * 2 ** 2,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.PixelShuffle(2),
                    activation(),
                ]
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SRResNet(nn.Module):
    """Super-Resolution Residual Neural Network
    https://arxiv.org/pdf/1609.04802v5.pdf
    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.
    channels: int
        Number of input and output channels
    num_blocks: int
        Number of stacked residual blocks
    """

    def __init__(
        self, scale_factor: int, channels: int = 3, num_blocks: int = 700
    ):
        super().__init__()

        # Pre Residual Blocks
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.PReLU(),
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=64, kernel_size=3, activation=nn.PReLU)
            for _ in range(num_blocks)
        ]
        self.res_blocks.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.res_blocks.append(nn.BatchNorm2d(num_features=64))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Upsamples
        n_upsamples = 1 if scale_factor == 2 else 2
        self.upsample = UpsampleBlock(
            n_upsamples=n_upsamples,
            channels=64,
            kernel_size=3,
            activation=nn.PReLU,
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                64, 3 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.PixelShuffle(scale_factor),
        )

        # Output layer
        self.tail = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=5 // 2,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=channels,
                kernel_size=9,
                stride=1,
                padding=9 // 2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve Low-Resolution input tensor
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor
        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor
        """
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x
