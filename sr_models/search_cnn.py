""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from sr_models.search_cells import SearchArch
import genotypes as gt
import logging


class SearchCNNController(nn.Module):
    """
    Stores and handles usage of alphas.
    Uses SearchArch from 'search_cells.py' as supernet and exapnds it with alphas for operations importance.
    """
    def __init__(
        self,
        c_init,
        c_fixed,
        bits,
        scale,
        arch_pattern,
        body_cells=2,
        device_ids=None,
        alpha_selector="softmax",
        quant_noise=False,
        skip_mode=True,
        primitives=None,
    ):
        super().__init__()
        self.body_cells = body_cells
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.temp = 1
        self.bits = bits
        self.primitives = primitives if not primitives is None else gt.PRIMITIVES_SR
        # initialize architect parameters: alphass
        self.n_ops = len(self.primitives)

        self.alphas_names = ["head", "body", "skip", "upsample", "tail"]

        self.alphas = dict()
        for name in self.alphas_names:
            params = nn.ParameterList()
            for _ in range(arch_pattern[name]):
                params.append(
                    nn.Parameter(torch.zeros(len(bits) * len(self.primitives[name])))
                )
            self.alphas[name] = params

        self.alphaselector = alphaSelector(name=alpha_selector)
        self.softmax = alphaSelector(name="softmax")

        # setup alphass list
        self._alphas = nn.ParameterList()
        for name in self.alphas:
            for p in self.alphas[name].parameters():
                self._alphas.append(p)

        self.net = SearchArch(
            c_init,
            c_fixed,
            bits,
            scale,
            arch_pattern,
            body_cells,
            quant_noise=quant_noise,
            skip_mode=skip_mode,
            primitives=self.primitives,
        )

    def get_alphas(self, func):
        alphas_projected = dict()
        for name in self.alphas_names:
            alphas_projected[name] = [
                func(alphas, self.temp, dim=-1) for alphas in self.alphas[name]
            ]
        return alphas_projected

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

    def get_max_alphas(self):
        return self.get_alphas(self.get_max)

    def forward_current_best(self, x):
        weight_alphass = self.get_alphas(self.get_max)
        return self.net(x, weight_alphass)

    def print_alphas(self, logger, temperature, writer, epoch):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### alphas #######")
        weight_alphas = self.get_alphas(self.softmax)
        for name in weight_alphas:
            logger.info(f"# alphas - {name}")
            for i, alphas in enumerate(weight_alphas[name]):
                logger.info(alphas)
                alpha_names = []
                for op_name in self.primitives[name]:
                    for bit in self.bits:
                        alpha_names += [f"{op_name}_{bit}"]
                assert len(alpha_names) == len(alphas.detach().cpu().numpy().tolist())
                writer.add_scalars(
                    f"alphas_softmax/{name}.{i}",
                    dict(zip(alpha_names, alphas.detach().cpu().numpy().tolist())),
                    epoch,
                )

        logger.info("# alphas_original #")
        for name in self.alphas:
            logger.info(f"# alphas - {name}")
            for i, alphas in enumerate(self.alphas[name]):
                logger.info(alphas)
                alpha_names = []
                for op_name in self.primitives[name]:
                    for bit in self.bits:
                        alpha_names += [f"{op_name}_{bit}"]
                assert len(alpha_names) == len(alphas.detach().cpu().numpy().tolist())
                writer.add_scalars(
                    f"alphas_orig/{name}.{i}",
                    dict(zip(alpha_names, alphas.detach().cpu().numpy().tolist())),
                    epoch,
                )

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene = dict()
        for name in self.alphas_names:
            gene[name] = gt.parse_sr(
                self.alphas[name], name, self.bits, self.primitives
            )
        return gt.Genotype_SR(**gene)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas_weights(self):
        for p in self._alphas:
            yield p

    # def named_alphass(self):
    #     for n, p in self._alphass:
    #         yield n, p

    def fetch_weighted_flops_and_memory(
        self,
    ):

        return self.net.fetch_weighted_flops_and_memory(
            self.get_alphas(self.softmax)
        )

    def get_max(self, alphas, temp=1, dim=-1, keep_weight=False):
        # get ones on the place of max values
        # alphas is 1d vector here
        values = alphas.max()
        ones = (values == alphas).type(torch.float32)

        if keep_weight:
            return alphas * ones.detach()
        else:
            return ones.detach()

    def fetch_current_best_flops_and_memory(self):
        return self.net.fetch_weighted_flops_and_memory(
            self.get_alphas(self.get_max)
        )


class alphaSelector:
    def __init__(self, name="softmax"):
        assert name in ["softmax", "gumbel", "hard_gumbel"]
        self.name = name

    def __call__(self, vector, temperature=1, dim=0):

        if self.name == "gumbel":
            return F.gumbel_softmax(vector, temperature, hard=False)

        if self.name == "hard_gumbel":
            return F.gumbel_softmax(vector, temperature, hard=True)

        if self.name == "softmax":
            return F.softmax(vector, dim)