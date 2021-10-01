""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models.search_cells import SearchArch
import genotypes as gt
import logging

# Weighted Soft Edge in forward
# Gumbel Final flops are not correct

from models.gumbel_top2 import gumbel_top2k


class SearchCNN(nn.Module):
    def __init__(self, n_nodes, c_in, repeat_factor, num_blocks):
        super().__init__()

        self.repeat_factor = repeat_factor
        self.net = nn.ModuleList()
        self.cnn_out = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True), nn.ReLU()
        )

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2)))
        )

        for i in range(num_blocks):
            self.net.append(
                SearchArch(n_nodes, c_in, repeat_factor, first=i == 0)
            )

    def forward(self, x, weight_alphas):

        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
        first_state = self.pixelup(state_zero)

        for block in self.net:
            x = block(x, weight_alphas)
        return self.cnn_out(x + first_state)

    def fetch_weighted_flops_and_memory(self, weight_alpha):
        flop = 0
        mem = 0
        for block in self.net:
            f, m = block.fetch_weighted_flops_and_memory(weight_alpha)
            flop += f
            mem += m
        return flop, mem


class SearchCNNController(nn.Module):
    """SearchCNN controller supporting multi-gpu"""

    def __init__(
        self,
        c_in,
        repeat_factor,
        criterion,
        n_nodes=4,
        device_ids=None,
        alpha_selector="softmax",
        blocks=2
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        self.n_ops = len(gt.PRIMITIVES_SR)

        self.alpha = nn.ParameterList()

        self.alphaselector = AlphaSelector(name=alpha_selector)
        self.softmax = AlphaSelector(name="softmax")

        for i in range(n_nodes):
            self.alpha.append(nn.Parameter(torch.ones(self.n_ops) / self.n_ops))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(n_nodes, c_in, repeat_factor, blocks)

    def forward(self, x, temperature=1, stable=False):

        if stable:
            func = self.softmax
        else:
            func = self.alphaselector

        weight_alphas = [
            func(alpha, temperature, dim=-1) for alpha in self.alpha
        ]

        out = self.net(x, weight_alphas)
        (flops, mem) = self.net.fetch_weighted_flops_and_memory(weight_alphas)
        return out, (flops, mem)

    def forward_current_best(self, x):
        weight_alphas = [self.get_max(a) for i, a in enumerate(self.alpha)]

        return self.net(x, weight_alphas)

    def print_alphas(self, logger, temperature):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha:
            logger.info(self.alphaselector(alpha, temperature, dim=-1))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene = gt.parse_sr(self.alpha)
        out = range(self.n_nodes, 2 + self.n_nodes)  # concat last two nodes
        return gt.Genotype_SR(normal=gene, normal_concat=out)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def fetch_weighted_flops_and_memory(
        self,
    ):

        return self.net.fetch_weighted_flops_and_memory(
            self.get_weights_normal(F.softmax)
        )

    def get_weights_normal(self, FN):
        return [FN(a) for a in self.alpha]

    def get_max(self, alpha, keep_weight=False):
        # get ones on the place of max values
        # alpha is 1d vector here
        values = alpha.max()
        ones = (values == alpha).type(torch.int)

        if keep_weight:
            return alpha * ones.detach()
        else:
            return ones.detach()

    def fetch_current_best_flops_and_memory(self):
        return self.net.fetch_weighted_flops_and_memory(
            self.get_weights_normal(self.get_max)
        )


class AlphaSelector:
    def __init__(self, name="softmax", use_soft_edge=False):
        assert name in ["softmax", "gumbel", "gumbel2k"]
        self.name = name

    def __call__(self, vector, temperature=1, dim=-1):

        if self.name == "gumbel":
            return F.gumbel_softmax(vector, temperature, hard=False)

        if self.name == "softmax":
            return F.softmax(vector, dim)

        if self.name == "gumbel2k":
            return gumbel_top2k(vector, temperature, dim)