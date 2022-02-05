# Quant module adapted from:
# https://github.com/zhutmost/lsq-net

import torch
import torch.nn as nn

""" HWGQ """


class _hwgq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return nn.functional.relu(x)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)


hwgq_steps = {
    1: 1.0576462792297525,
    2: 0.6356366866203315,
    3: 0.3720645813370479,
    4: 0.21305606790772952,
    8: 0.020300567823662602,
    16: 9.714825915156693e-05,
}


"""  LSQ  """


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(nn.Module):
    def __init__(
        self, bit, all_positive=False, symmetric=False, per_channel=True
    ):
        super(LsqQuan, self).__init__()
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)
                * 2
                / (self.thd_pos ** 0.5)
            )
        else:
            self.s = nn.Parameter(
                x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            )

    def forward(self, x):
        if self.bit >= 32:
            return x

        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        device = x.device
        s_scale = grad_scale(self.s, s_grad_scale).to(device)
        x = x / (s_scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * (s_scale)
        return x


class QuantConv(nn.Conv2d):

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

    def __init__(self, **kwargs):
        super(QuantConv, self).__init__(**kwargs)
        self.kernel = self.to_tuple(self.kernel_size)
        self.stride = self.to_tuple(self.stride)
        self.padding = self.to_tuple(self.padding)
        self.dilation = self.to_tuple(self.dilation)
        # complexities
        # FLOPs = 2 x Cin x Cout x k**2 x Wout x Hout / groups
        self.param_size = (
            2
            * self.in_channels
            * self.out_channels
            * self.kernel[0]
            * self.kernel[1]
            / self.groups
        )  # * 1e-6  # stil unsure why we use 1e-6
        self.register_buffer("flops", torch.tensor(0, dtype=torch.float))
        self.register_buffer("memory_size", torch.tensor(0, dtype=torch.float))

    def to_tuple(self, value):
        if type(value) == int:
            return (value, value)
        if type(value) == tuple:
            return value

    def forward(self, input_x, quantized_weight):
        """
        BATCH x C x W x H

        """
        # get the same device to avoid errors
        device = input_x.device

        c_in, w_in, h_in = input_x.shape[1], input_x.shape[2], input_x.shape[3]

        w_out = self.compute_out(w_in, "w")
        h_out = self.compute_out(h_in, "h")

        tmp = torch.tensor(c_in * w_in * h_in, dtype=torch.float).to(device)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(
            self.param_size * w_out * h_out, dtype=torch.float
        ).to(device)
        self.flops.copy_(tmp)
        del tmp

        return self._conv_forward(input_x, quantized_weight, bias=None)

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

    def _fetch_info(self):
        return self.flops.item(), self.memory_size.item()


# USE Instead of CNN + ReLU Block for final quantized model
class QAConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(QAConv2d, self).__init__()
        self.bit = kwargs.pop("bits")[0]
        self.bit_orig = self.bit
        self.conv = QuantConv(**kwargs)
        self._set_q_fucntions(self.bit)

    # can be used to modify bits during the training
    def _set_q_fucntions(self, bit):
        self.quan_a_fn = HWGQ(bit)
        self.quan_w_fn = LsqQuan(
            bit, all_positive=False, symmetric=False, per_channel=True
        )
        self.quan_w_fn.init_from(self.conv.weight)

    def set_fp(self, bit=32):
        self._set_q_fucntions(bit)

    def set_quant(self):
        self._set_q_fucntions(self.bit)

    def forward(self, input_x):
        quantized_weight = self.quan_w_fn(self.conv.weight)
        quantized_act = self.quan_a_fn(input_x)
        out = self.conv(quantized_act, quantized_weight)
        return out

    def _fetch_info(self):
        f, m = self.conv._fetch_info()
        return f * self.bit, m * self.bit


class SepQAConv2d(nn.Module):
    def __init__(self, **kwargs):
        super(SepQAConv2d, self).__init__()
        self.bits = kwargs.pop("bits")
        self.conv = nn.ModuleList()
        print('Using SepQA')
        for _ in range(len(self.bits)):
            self.conv.append(QuantConv(**kwargs))

        self.acts = [HWGQ(bit) for bit in self.bits]
        self.q_fn = []
        for bit in self.bits:
            self.q_fn.append(
                LsqQuan(
                    bit, all_positive=False, symmetric=False, per_channel=True
                )
            )
            self.q_fn[-1].init_from(self.conv[-1].weight)

        self.alphas = [1] * len(self.bits)

    def forward(self, input_x):
        out = []
        for alpha, act, conv, q_fn in zip(self.alphas, self.acts, self.conv, self.q_fn):
            weights = q_fn(conv.weight)
            acts = act(input_x)
            out.append(alpha*conv(acts, weights))
        return sum(out)

    def _fetch_info(self):
        bit_ops, mem = 0, 0
        b, m = self.conv[-1]._fetch_info()

        for bit, alpha in zip(self.bits, self.alphas):
            bit_ops += alpha * b * bit
            mem += alpha * m * bit
        return bit_ops, mem

    def set_alphas(self, alphas):
        self.alphas = alphas


class SharedQAConv2d(nn.Module):
    def __init__(self, **kwargs):
        super(SharedQAConv2d, self).__init__()
        self.bits = kwargs.pop("bits")
        self.conv = QuantConv(**kwargs)
        self.acts = [HWGQ(bit) for bit in self.bits]
        self.q_fn = []
        for bit in self.bits:
            self.q_fn.append(
                LsqQuan(
                    bit, all_positive=False, symmetric=False, per_channel=True
                )
            )
            self.q_fn[-1].init_from(self.conv.weight)

        self.alphas = [1] * len(self.bits)

    def forward(self, input_x):
        weights = torch.zeros_like(self.conv.weight)
        acts = torch.zeros_like(input_x)
        alphas = self.alphas / self.alphas.sum()
        for alpha, act, q_fn in zip(alphas, self.acts, self.q_fn):
            weights += alpha * q_fn(self.conv.weight)
            acts += alpha * act(input_x)

        return self.conv(acts, weights)

    def _fetch_info(self):
        bit_ops, mem = 0, 0
        b, m = self.conv._fetch_info()

        for bit, alpha in zip(self.bits, self.alphas):
            bit_ops += alpha * b * bit
            mem += alpha * m * bit
        return bit_ops, mem

    def set_alphas(self, alphas):
        self.alphas = alphas


class BaseConv(nn.Module):
    def __init__(self, *args, **kwargs):
        shared = kwargs.pop("shared")
        super(BaseConv, self).__init__()
        if shared:
            self.conv_func = SharedQAConv2d
        elif shared is False:
            self.conv_func = QAConv2d
        else:
            self.conv_func = SepQAConv2d
        self.alphas = 1

    def set_alphas(self, alphas):
        for m in self.modules():
            if isinstance(m, self.conv_func):
                m.set_alphas(alphas)
        self.alphas = alphas

    def fetch_info(self):
        sum_flops = 0
        sum_memory = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                f, mem = m._fetch_info()
                sum_flops += f
                sum_memory += mem

        return sum_flops, sum_memory
