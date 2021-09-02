""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops_flops as ops


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


def mean_by_c(image, repeat_factor):
    b, d, w, h = image.shape
    image = image.reshape([b, d // repeat_factor, repeat_factor, w, h])
    return image.mean(2)


class SearchArch(nn.Module):
    """Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, c_init, repeat_factor):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_fixed: # of channels to work with
            C_init:  # of initial channels, usually 3
        """

        super().__init__()
        self.n_nodes = n_nodes
        self.c_fixed = c_init ** repeat_factor
        self.repeat_factor = repeat_factor
        self.c_init = c_init

        # Used for soft edge experiments to stabilize training after warm up
        self.edge_n = nn.ParameterList()
        self.edge_r = nn.ParameterList()

        for i in range(n_nodes):
            self.edge_n.append(nn.Parameter(torch.ones(i + 1)))
            self.edge_r.append(nn.Parameter(torch.ones(i + 1)))

        assert is_square(repeat_factor), "Repear factor should be a square of N"

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(1 + i):  # include 1 input nodes
                op = ops.MixedOp(self.c_fixed, 1)
                self.dag[i].append(op)

    def assertion_in(self, size_in):
        assert (
            size_in[1] == self.channels
        ), f"Input size {size_in}, does not match fixed channels {self.c_init}, called from {self.func_name}"

    def forward(self, x, w_dag):
        self.assertion_in(x.shape)
        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)

        states = [state_zero]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(
                edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list))
            )
            states.append(s_cur)

        # Only two last states
        s_out = torch.cat(states[:-2], dim=1).sum(1)
        out = torch.nn.functional.pixel_unshuffle(
            s_out, int(self.repeat_factor ** (1 / 2))
        )
        # Final output should have the same number of channels as input
        self.assertion_in(out.shape)
        return out

    def fetch_weighted_flops_and_memory(self, w_dag):
        total_flops = 0
        total_memory = 0

        for edges, w_list in zip(self.dag, w_dag):
            total_flops += sum(
                edges[i].fetch_weighted_flops(w) for i, w in enumerate(w_list)
            )
            total_memory += sum(
                edges[i].fetch_weighted_memory(w) for i, w in enumerate(w_list)
            )

        return total_flops, total_memory
