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
        c_in,
        repeat_factor,
        criterion,
        n_nodes=4,
        device_ids=None,
        use_soft_edge=False,
        alpha_selector="softmax",
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

        self.alphaselector = AlphaSelector(
            name=alpha_selector, use_soft_edge=use_soft_edge
        )
        self.softmax = AlphaSelector(
            name="softmax", use_soft_edge=use_soft_edge
        )

        self.use_soft_edge = use_soft_edge

        self.alpha.append(nn.Parameter(torch.ones(1, self.n_ops) / self.n_ops))
        for i in range(n_nodes - 1):
            self.alpha.append(
                nn.Parameter(torch.ones(i + 1, self.n_ops) / self.n_ops)
            )

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchArch(n_nodes, c_in, repeat_factor)

    def forward(self, x, temperature=1, stable=False):

        if stable:
            func = self.softmax
        else:
            func = self.alphaselector

        weight_alphas = [
            func(alpha, edge_w, temperature, dim=-1)
            for alpha, edge_w in zip(self.alpha, self.net.edge_n)
        ]

        out = self.net(x, weight_alphas)
        (flops, mem) = self.net.fetch_weighted_flops_and_memory(weight_alphas)
        return out, (flops, mem)

    def forward_current_best(self, x):

        weight_alphas = [
            self.get_max(self.prod(a, self.net.edge_n[i]))
            for i, a in enumerate(self.alpha)
        ]

        return self.net(x, weight_alphas)

    def print_alphas(self, logger, temperature):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha, edge in zip(self.alpha, self.net.edge_n):
            logger.info(self.alphaselector(alpha, edge, temperature, dim=-1))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def print_edges(self, logger):

        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### EDGES #######")
        logger.info("# EDGE W  - normal")
        for edge in self.net.edge_n:
            logger.info(F.softmax(edge, dim=-1))

        logger.info("\n# EDGE W - reduce")
        for edge in self.net.edge_r:
            logger.info(F.softmax(edge, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        alpha_normal = [
            self.prod(a, self.net.edge_n[i]) for i, a in enumerate(self.alpha)
        ]

        gene = gt.parse_sr(alpha_normal, k=2)
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
        return [
            FN(self.prod(a, self.net.edge_n[i]))
            for i, a in enumerate(self.alpha)
        ]

    def prod(self, vector, edge):
        if self.use_soft_edge:
            return (vector.T * F.softmax(edge)).T

        else:
            return vector

    def get_max(self, alpha, k=2, keep_weight=False):
        values, indices = alpha[:, :-1].max(1)
        ones = (values.unsqueeze(1) == alpha).type(torch.int)
        zero_rows = [
            i
            for i in range(alpha.shape[0])
            if not i in values.topk(min(k, alpha.shape[0])).indices
        ]
        ones[zero_rows] = 0
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
        self.use_soft_edge = use_soft_edge

    def prod(self, vector, edge):
        if self.use_soft_edge:
            return (vector.T * F.softmax(edge)).T

        else:
            return vector

    def __call__(self, vector, edge, temperature=1, dim=-1):

        if self.name == "gumbel":
            return self.prod(F.gumbel_softmax(vector, temperature,  hard=False), edge)

        if self.name == "softmax":
            return self.prod(F.softmax(vector, dim), edge)

        if self.name == "gumbel2k":
            return self.prod(gumbel_top2k(vector, temperature, dim), edge)
