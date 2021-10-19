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

        # initialize architect parameters: alphass
        self.n_ops = len(gt.PRIMITIVES_SR)

        self.alphas_names = ["head", "body", "skip", "upsample", "tail"]

        self.alphas = dict()
        for name in self.alphas_names:
            params = nn.ParameterList()
            for _ in range(arch_pattern[name]):
                params.append(
                    nn.Parameter(torch.ones(len(gt.PRIMITIVES_SR[name])))
                )
            self.alphas[name] = params

        self.alphaselector = alphaSelector(name=alpha_selector)
        self.softmax = alphaSelector(name="softmax")

        # setup alphass list
        self._alphas = []
        for name in self.alphas:
            for n, p in self.alphas[name].named_parameters():
                self._alphas.append((n, p))

        self.net = SearchArch(c_init, c_fixed, scale, arch_pattern, body_cells)

    def get_alphass(self, func):
        alphass_projected = dict()
        for name in self.alphas_names:
            alphass_projected[name] = [
                func(alphas, self.temp, dim=-1) for alphas in self.alphas[name]
            ]

    def forward(self, x, temperature=1, stable=False):
        self.temp = temperature

        if stable:
            func = self.softmax
        else:
            func = self.alphaselector

        weight_alphass = self.get_alphass(func)

        out = self.net(x, weight_alphass)
        (flops, mem) = self.net.fetch_weighted_flops_and_memory(weight_alphass)
        return out, (flops, mem)

    def forward_current_best(self, x):
        weight_alphass = self.get_alphass(self.get_max)
        return self.net(x, weight_alphass)

    def print_alphass(self, logger, temperature):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### alphas #######")
        weight_alphass = self.get_alphass(self.alphaselector, temperature)
        for name in weight_alphass:
            logger.info(f"# alphas - {name}")
            for alphas in weight_alphass[name]:
                logger.info(alphas)

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene = dict()
        for name in self.alphas_names:
            gene[name] = gt.parse_sr(self.alphas[name], name)
        return gt.Genotype_SR(**gene)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas_weights(self):
        for n, p in self._alphas:
            yield p

    def named_alphass(self):
        for n, p in self._alphass:
            yield n, p

    def fetch_weighted_flops_and_memory(
        self,
    ):

        return self.net.fetch_weighted_flops_and_memory(
            self.self.get_alphass(F.softmax, self.temp)
        )

    def get_max(self, alphas, keep_weight=False):
        # get ones on the place of max values
        # alphas is 1d vector here
        values = alphas.max()
        ones = (values == alphas).type(torch.int)

        if keep_weight:
            return alphas * ones.detach()
        else:
            return ones.detach()

    def fetch_current_best_flops_and_memory(self):
        return self.net.fetch_weighted_flops_and_memory(
            self.self.get_alphass(self.get_max, self.temp)
        )


class alphaSelector:
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