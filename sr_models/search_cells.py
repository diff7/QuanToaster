""" CNN cell for architecture search """
import torch
import torch.nn as nn
from sr_models import ops_flops as ops


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


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
        self.c_fixed = c_init * repeat_factor
        self.repeat_factor = repeat_factor
        self.c_init = c_init

        # Used for soft edge experiments to stabilize training after warm up
        self.edge_n = nn.ParameterList()
        self.edge_r = nn.ParameterList()

        self.edge_n.append(nn.Parameter(torch.ones(1)))
        self.edge_r.append(nn.Parameter(torch.ones(1)))
        for i in range(n_nodes - 1):
            self.edge_n.append(nn.Parameter(torch.ones(i + 1)))
            self.edge_r.append(nn.Parameter(torch.ones(i + 1)))

        assert is_square(repeat_factor), "Repear factor should be a square of N"

        # generate dag
        self.dag = nn.ModuleList()
        print("INITIALIZING MODEL's DAG")
        self.dag.append(nn.ModuleList())
        self.dag[0].append(ops.MixedOp(self.c_fixed, 1))
        for i in range(self.n_nodes - 1):
            self.dag.append(nn.ModuleList())
            for j in range(1 + i):  # include 1 input nodes
                print("initialized:", i, j, "C_FIXED", self.c_fixed)
                op = ops.MixedOp(self.c_fixed, 1)
                self.dag[i + 1].append(op)

        self.pixelup = nn.Sequential(
            nn.PixelShuffle(int(repeat_factor ** (1 / 2))), nn.PReLU()
        )

    def assertion_in(self, size_in):
        assert int(size_in[1]) == int(
            self.c_fixed
        ), f"Input size {size_in}, does not match fixed channels {self.c_fixed}"

    def forward(self, x, w_dag):
        state_zero = torch.repeat_interleave(x, self.repeat_factor, 1)
        self.assertion_in(state_zero.shape)

        s_cur = self.dag[0][0](state_zero, w_dag[0][0])
        states = [s_cur]
        for edges, w_list in zip(self.dag[1:], w_dag[1:]):
            s_cur = sum(
                edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list))
            )
            states.append(s_cur)

        self.assertion_in(s_cur.shape)

        out = self.pixelup(s_cur)
        x_residual = self.pixelup(state_zero)

        return out + x_residual

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


Genotype_SR(
    normal=[
        [("simple_1x1_grouped_full", 0)],
        [("growth2_5x5", 0)],
        [("conv_3x1_1x3_growth2", 1), ("simple_1x1_groupe_full", 0)],
    ],
    normal_concat=range(3, 5),
)
