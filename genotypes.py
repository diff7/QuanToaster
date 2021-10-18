""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
from sr_models import ops_flops as ops_sr
from models import ops_flops as ops_cls


Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
Genotype_SR = namedtuple("Genotype_SR", "head body tail skip upsample")


PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",  # identity
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "none",
]


PRIMITIVES_SR = [
    # "skip_connect",  # identity
    # "conv_5x1_1x5",
    # "conv_3x1_1x3",
    "simple_3x3",
    "simple_1x1",
    "simple_5x5",
    # "simple_1x1_grouped_full",
    # "simple_3x3_grouped_full",
    # "simple_5x5_grouped_full",
    # "simple_1x1_grouped_3",
    # "simple_3x3_grouped_3",
    # "simple_5x5_grouped_3",
    "DWS_3x3",
    "DWS_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "decenc_3x3_4",
    "decenc_3x3_2",
    "decenc_5x5_2",
    # "decenc_5x5_8",
    # "decenc_3x3_8",
    # "decenc_3x3_4_g3",
    # "decenc_3x3_2_g3",
    # "decenc_5x5_2_g3",
    # "decenc_5x5_8_g3",
    # "decenc_3x3_8_g3",
    # "growth4_3x3",
    # "bs_up_bicubic_residual",
    # "bs_up_nearest_residual",
    # "bs_up_bilinear_residual",
    # "bs_up_bicubic",
    # "bs_up_nearest",
    # "bs_up_bilinear",
    # "none",
]


def to_dag(C_in, gene, reduction):
    """generate discrete ops from gene"""
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops_cls.OPS[op_name](C_in, stride, True)
            if not isinstance(
                op, ops_cls.Identity
            ):  # Identity does not use drop path
                op = nn.Sequential(op, ops_cls.DropPath_())
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def to_dag_sr(C_fixed, gene, gene_type, c_in=3, c_out=3, scale=4):
    """generate discrete ops from gene"""
    dag = []
    for i, edges in enumerate(gene):
        C_in, C_out, = (
            C_fixed,
            C_fixed,
        )
        if i == 0 and gene_type == "head":
            C_in = c_in
        elif i + 1 == len(gene) and gene_type == "tail":
            C_out = c_out
        elif i == 0 and gene_type == "tail":
            C_out = c_out

        elif gene_type == "upsample":
            C_in = C_fixed
            C_out = 3 * (scale ** 2)
        else:
            C_in = C_fixed
            C_out = C_fixed

        for op_name, _ in edges:
            op = ops_sr.OPS[op_name](C_in, C_out, C_fixed, 1, True)
            dag.append(op)
    return nn.Sequential(*dag)


def from_str(s):
    """generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)
    return genotype


def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CN"N.
    """

    gene = []
    assert PRIMITIVES[-1] == "none"  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(
            edges[:, :-1], 1
        )  # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)

        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_sr(alpha):

    gene = []
    # assert PRIMITIVES_SR[-1] == "none"  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, edges in enumerate(alpha):

        func_idx = edges.argmax()  # ignore 'none'
        prim = PRIMITIVES_SR[func_idx]
        node_gene = [(prim, i)]
        gene.append(node_gene)

    return gene
