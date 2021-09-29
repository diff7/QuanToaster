import torch
import torch.nn as nn
from sr_models import ops_flops as ops_sr
from sr_models.flops import BaseConv


class ManualCNN(nn.Module):
    """Augmented CNN model"""

    def __init__(self, c_init, repeat_factor):
        super().__init__()

        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor

        self.cv1 = ops_sr.OPS["conv_3x1_1x3"](self.c_fixed, 1, True)
        self.cv2 = ops_sr.OPS["conv_3x1_1x3"](self.c_fixed, 1, True)
        self.cv3 = ops_sr.OPS["conv_3x1_1x3"](self.c_fixed, 1, True)
        self.cv4 = ops_sr.OPS["conv_3x1_1x3"](self.c_fixed, 1, True)
        self.cv5 = nn.Sequential(ops_sr.OPS["DWS_3x3"](self.c_fixed, 1, True), ops_sr.DropPath_()) 
        self.cv6 = nn.Sequential(ops_sr.OPS["conv_3x1_1x3"](self.c_fixed, 1, True), ops_sr.DropPath_()) 

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.ReLU()
        )

    def forward(self, x0):
        x0 = torch.repeat_interleave(x0, self.repeat_factor, 1)

        x1 = self.cv1(x0)
        x2 = self.cv2(x1)
        x3 = self.cv5(x2)
        x4 = self.cv4(x3)

        out = self.pixelup(x4)

        x_residual = self.cv6(x0)
        x_residual = self.pixelup(x_residual)
        return  out #+ x_residual

    def drop_path_prob(self, p):
        """Set drop path probability"""
        for module in self.modules():
            if isinstance(module, ops_sr.DropPath_):
                module.p = p


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


# BEST ARCH


# class FromGENE(nn.Module):
#     def __init__(self, c_init, repeat_factor):
#         super().__init__()

#         self.c_fixed = c_init * repeat_factor
#         self.repeat_factor = repeat_factor

#         self.c1 = ops_sr.OPS["decenc_3x3"](self.c_fixed, 1, True)
#         self.c2 = ops_sr.OPS["simple_5x5"](self.c_fixed, 1, True)
#         self.c3 = ops_sr.OPS["simple_3x3"](self.c_fixed, 1, True)
#         self.c4 = ops_sr.OPS["simple_3x3"](self.c_fixed, 1, True)
#         self.c5 = ops_sr.OPS["decenc_3x3"](self.c_fixed, 1, True)
#         self.c6 = ops_sr.OPS["simple_5x5"](self.c_fixed, 1, True)

#         self.pixelup = nn.Sequential(
#             nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
#         )

#     def forward(self, x0):
#         x0 = torch.repeat_interleave(x0, self.repeat_factor, 1)
#         x1 = self.c1(x0)
#         x2 = self.c2(x1)  # self.c3(x0)
#         x3 = self.c4(x2)  # self.c5(x1)
#         x4 = self.c6(x3)  # x0

#         s_out = x4 + x3

#         out = self.pixelup(s_out)
#         x_residual = self.pixelup(x0)
#         return out + x_residual


if __name__ == "__main__":
    random_image = torch.randn(3, 48, 32, 32)

    names = []
    flops = []
    model = ESPCN(4)
    model(random_image)

    f = model.fetch_info()[0]
    print(n, "FLOPS:", round(f, 5))