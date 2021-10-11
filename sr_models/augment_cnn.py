""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops
import genotypes as gt


# class SubPixelConvBlock(nn.Module):
#     def __init__(
#         self,
#         input_nc=3,
#         output_nc=3,
#         upscale_factor=2,
#         kernel_size=3,
#         stride=1,
#         bias=True,
#         pad_type="zero",
#         norm_type=None,
#         act_type="prelu",
#         use_dropout=False,
#     ):
#         super(SubPixelConvBlock, self).__init__()
#         self.conv_block = nn.Conv2d(
#             48, 48, kernel_size=3, padding=1, bias=False
#         )

#         self.PS = nn.PixelShuffle(upscale_factor)

#     def forward(self, input):
#         output = self.conv_block(input)
#         output = self.PS(output)
#         return output


class AugmentCNN(nn.Module):
    """Augmented CNN model"""

    def __init__(self, c_init, repeat_factor, genotype, blocks=4):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.c_fixed = 32  # c_init * repeat_factor
        self.repeat_factor = repeat_factor

        net = [gt.to_dag_sr(self.c_fixed, genotype.normal)] * blocks
        self.dag = nn.Sequential(*net)
        self.dag_len = len(self.dag)

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2)))
        )

        # self.pixelup = nn.Sequential(SubPixelConvBlock(), SubPixelConvBlock())

        self.space_to_depth = torch.nn.functional.pixel_unshuffle
        self.cnn_out = nn.Sequential(
            nn.Conv2d(self.c_fixed, 48, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
        )
        self.cnn_in = nn.Sequential(
            nn.Conv2d(3, self.c_fixed, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
        )

    def forward(self, x):
        # x = self.cnn_in(x)
        outs = []
        for i, block in enumerate(self.dag):
            if i == 0:
                # state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
                state_zero = self.cnn_in(x)
                self.assertion_in(state_zero.shape)
                first_state = state_zero  # self.pixelup(state_zero)

            else:
                state_zero = x

                # self.space_to_depth(
                #     x, int(self.repeat_factor ** (1 / 2))
                # )

            s_cur = state_zero
            ss = []
            for op in block[:-1]:
                s_cur = op(s_cur)
                ss.append(s_cur)
            s_skip = block[-1](state_zero)
            self.assertion_in(s_cur.shape)

            x = (s_cur + s_skip) * 0.2 + state_zero
            # res = self.pixelup()

            # x = out + res
            outs.append(x)

        x = self.cnn_out(x + first_state)
        x = self.pixelup(x)
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
