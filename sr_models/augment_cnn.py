""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models import ops_flops as ops
import genotypes as gt


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

        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor

        net = [gt.to_dag_sr(self.c_fixed, genotype.normal)] * blocks
        self.dag = nn.Sequential(*net)
        self.dag_len = len(self.dag)

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
        )
        self.space_to_depth = torch.nn.functional.pixel_unshuffle
        self.cnn_out = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False), nn.ReLU()
        )

        # self.skip_cnn = nn.ModuleList()

        # for _ in range(blocks):
        #     self.skip_cnn.append(
        #         nn.Sequential(
        #             nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
        #             nn.ReLU(),
        #         )
        #     )

    def forward(self, x):
        for i, block in enumerate(self.dag):
            if i == 0:
                state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
                self.assertion_in(state_zero.shape)
                first_state = self.pixelup(state_zero)

            else:
                state_zero = self.space_to_depth(
                    x, int(self.repeat_factor ** (1 / 2))
                )

            s_cur = state_zero
            for op in block[:-1]:
                s_cur = op(s_cur)

            s_skip = block[-1](state_zero)
            self.assertion_in(s_cur.shape)

            out = self.pixelup(s_cur)
            x_residual = self.pixelup(s_skip)

            x = (out + x_residual)/2
        return self.cnn_out(x + first_state)

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
