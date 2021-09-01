import torch
import torch.nn as nn


class ConvFlops(nn.Module):

    """
    Flops are computed for square kernel
    FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
    We use 2 because 1 for multiplocation and 1 for addition

    Hout = Hin + 2*padding[0] - dilation[0] x (kernel[0]-1)-1
          --------------------------------------------------- + 1
                                stride
    Wout same as above


    NOTE: We do not account for bias term

    """

    def __init__(
        self,
        C_in,
        C_out,
        kernel,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):

        super(ConvFlops, self).__init__()
        self.conv = nn.Conv2d(
            C_in,
            C_out,
            kernel,
            stride,
            padding=0,
            dilation=1,
            groups=groups,
            bias=False,
        )

        self.kernel = self.to_tuple(kernel)
        self.stride = self.to_tuple(stride)
        self.padding = self.to_tuple(padding)
        self.dilation = self.to_tuple(dilation)
        self.groups = groups

        # complexities
        # FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
        self.param_size = (
            2 * C_in * C_out * self.kernel[0] * self.kernel[1] / self.groups
        )  # * 1e-6  # stil unsure why we use 1e-6
        self.register_buffer("flops", torch.tensor(0, dtype=torch.float))
        self.register_buffer("memory_size", torch.tensor(0, dtype=torch.float))

    def to_tuple(self, value):
        if type(value) == int:
            return (value, value)
        if type(value) == tuple:
            return value

    def forward(self, input):
        """
        BATCH x C x W x H

        """
        c_in, w_in, h_in = input.shape[1], input.shape[2], input.shape[3]

        w_out = self.compute_out(w_in, "w")
        h_out = self.compute_out(h_in, "h")

        tmp = torch.tensor(c_in * w_in * h_in, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.param_size * w_out * h_out, dtype=torch.float)
        self.flops.copy_(tmp)
        del tmp
        out = self.conv(input)
        return out

    def compute_out(self, input_size, spatial="w"):

        if spatial == "w":
            idx = 0
        if spatial == "h":
            idx = 1
        return int(
            (
                input_size
                + 2 * self.padding[idx]
                - self.dilation[idx] * (self.kernel[idx] - 1)
                - 1
            )
            / self.stride[idx]
            + 1
        )


class BaseConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseConv, self).__init__()
        self.conv_func = ConvFlops

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                sum_flops += m.flops.item()
                sum_memory += m.memory_size.item()

        return torch.tensor([sum_flops]), torch.tensor([sum_memory])


# TODO: verify the accuracy
def count_upsample_flops(mode, shape):

    shape_product = torch.prod(torch.tensor(shape)[1:])
    """
    Source ::
    https://github.com/Lyken17/pytorch-OpCounter/

    """

    if mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        print(f"Flops count for {mode} upsample is not implemented")
        return torch.tensor([0])

    if mode == "nearest":
        return torch.tensor([0])

    if mode == "linear":
        total_ops = shape_product * 5  # 2 muls + 3 add
    elif mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = shape_product * 11  # 6 muls + 5 adds
    elif mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = shape_product * (ops_solve_A + ops_solve_p)
    elif mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = shape_product * (13 * 2 + 5)

    return torch.tensor([total_ops])
