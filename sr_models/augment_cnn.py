""" CNN for network augmentation """
import torch
import torch.nn as nn
from sr_models import ops_flops as ops
import genotypes as gt


class AugmentCNN(nn.Module):
    """ Augmented CNN model """

    def __init__(self, c_init, repeat_factor, genotype):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.genotype = genotype

        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor

        self.dag = gt.to_dag_sr(self.c_fixed, genotype)

    def forward(self, x):

        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
        self.assertion_in(state_zero.shape)

        states = [state_zero]

        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = states[-1] + states[-2]
        self.assertion_in(s_out.shape)
        out = torch.nn.functional.pixel_shuffle(
            s_out, int(self.repeat_factor ** (1 / 2))
        )
        return out

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p

    def assertion_in(self, size_in):
        assert int(size_in[1]) == int(
            self.c_fixed
        ), f"Input size {size_in}, does not match fixed channels {self.c_fixed}"