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


class SearchCNNController(nn.Module):
    """SearchCNN controller supporting multi-gpu"""

    def __init__(
        self,
        c_init,
        c_fixed,
        scale,
        criterion,
        arch_pattern,
        body_cells=2,
        device_ids=None,
        alpha_selector="softmax",
    ):
        super().__init__()
        self.body_cells = body_cells
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        self.n_ops = len(gt.PRIMITIVES_SR)

        self.alpha_names = ["head", "body", "skip", "upsample", "tail"]

        self.alpha = nn.ParameterDict()
        for name in self.alpha_names:
            params = nn.ParameterList()
            for _ in range(arch_pattern[name]):
                params.append(
                    nn.Parameter(torch.ones(len(gt.PRIMITIVES_SR[name])))
                )
            self.alpha[name] = params

        self.alphaselector = AlphaSelector(name=alpha_selector)
        self.softmax = AlphaSelector(name="softmax")

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchArch(c_init, scale, c_fixed, arch_pattern, body_cells)

    def get_alphas(self, func):
        alphas_projected = nn.ParameterDict()
        for name in self.alpha_names:
            alphas_projected[name] = [
                func(alpha, self.temp, dim=-1) for alpha in self.alpha[name]
            ]

    def forward(self, x, temperature=1, stable=False):
        self.temp = temperature

        if stable:
            func = self.softmax
        else:
            func = self.alphaselector

        weight_alphas = self.get_alphas(func)

        out = self.net(x, weight_alphas)
        (flops, mem) = self.net.fetch_weighted_flops_and_memory(weight_alphas)
        return out, (flops, mem)

    def forward_current_best(self, x):
        weight_alphas = self.get_alphas(self.get_max)
        return self.net(x, weight_alphas)

    def print_alphas(self, logger, temperature):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        weight_alphas = self.get_alphas(self.alphaselector, temperature)
        for name in weight_alphas:
            logger.info(f"# Alpha - {name}")
            for alpha in weight_alphas[name]:
                logger.info(alpha)

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene = dict()
        for name in self.alpha_names:
            gene[name] = gt.parse_sr(self.alpha[name], name)
        return gt.Genotype_SR(**gene)

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
            self.self.get_alphas(F.softmax, self.temp)
        )

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
            self.self.get_alphas(self.get_max, self.temp)
        )


class AlphaSelector:
    def __init__(self, name="softmax"):
        assert name in ["softmax", "gumbel", "gumbel2k"]
        self.name = name

    def __call__(self, vector, temperature=1, dim=-1):

        if self.name == "gumbel":
            return F.gumbel_softmax(vector, temperature, hard=False)

        if self.name == "softmax":
            return F.softmax(vector, dim)

        if self.name == "gumbel2k":
            return gumbel_top2k(vector, temperature, dim)