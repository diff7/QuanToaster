""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch.nn as nn
from sr_models import quant_ops as ops_sr

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


body = [
    #    "skip_connect",
    "conv_5x1_1x5",
    "conv_3x1_1x3",
    "simple_3x3",
    "simple_1x1",
    "simple_5x5",
    # "simple_1x1_grouped_full",
    # "simple_3x3_grouped_full",
    # "simple_5x5_grouped_full",
    "simple_1x1_grouped_3",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
    "DWS_3x3",
    "DWS_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "decenc_3x3_4",
    "decenc_3x3_2",
    "decenc_5x5_2",
    "decenc_5x5_8",
    "decenc_3x3_8",
    # "decenc_3x3_4_g3",
    # "decenc_3x3_2_g3",
    # "decenc_5x5_2_g3",
]
head = [
    # "skip_connect",
    "conv_5x1_1x5",
    "conv_3x1_1x3",
    "simple_3x3",
    "simple_1x1",
    "simple_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "simple_1x1_grouped_3",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
]
PRIMITIVES_SR = {
    "head": head,
    "body": body,
    "skip": body,
    "tail": head,
    "upsample": head,
}


def from_str(s):
    genotype = eval(s)
    return genotype


def to_dag_sr(C_fixed, gene, gene_type, c_in=3, c_out=3, scale=4):
    """generate discrete ops from gene"""
    dag = []
    for i, (op_name, bit) in enumerate(gene):
        C_in, C_out, = (
            C_fixed,
            C_fixed,
        )
        if i == 0 and gene_type == "head":
            C_in = c_in
        elif i + 1 == len(gene) and gene_type == "tail":
            C_out = c_out
        elif i == 0 and gene_type == "tail":
            C_in = c_in

        elif gene_type == "upsample":
            C_in = C_fixed
            C_out = 3 * (scale ** 2)
        else:
            C_in = C_fixed
            C_out = C_fixed

        print(gene_type, op_name, C_in, C_out, C_fixed, bit)
        op = ops_sr.OPS[op_name](
            C_in, C_out, [bit], C_fixed, 1, False, shared=False
        )
        dag.append(op)
    return nn.Sequential(*dag)


def parse_sr(alpha, name):

    gene = []

    for i, edges in enumerate(alpha):
        func_idx = edges.argmax()  # ignore 'none'
        prim = PRIMITIVES_SR[name][func_idx]
        gene.append(prim)

    return gene
