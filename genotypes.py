"""
Specify detailed search space for the architechture.
"""
import torch.nn as nn
from collections import namedtuple
from sr_models import quant_ops as ops_sr

Genotype_SR = namedtuple("Genotype_SR", "head body tail skip upsample")


body = [
    "skip_connect",
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
    "decenc_3x3_4_g3",
    "decenc_3x3_2_g3",
    "decenc_5x5_2_g3",
]

head = [
    "conv_5x1_1x5",
    "conv_3x1_1x3",
    "simple_1x1",
    "simple_3x3",
    "simple_5x5",
    "growth2_5x5",
    "growth2_3x3",
    "simple_1x1_grouped_3",
    "simple_3x3_grouped_3",
    "simple_5x5_grouped_3",
]

tail = [
    'simple_1x1',
    "simple_3x3",
    "simple_5x5",
    # "growth2_5x5",
    # "growth2_3x3",
    # "simple_3x3_grouped_3",
    # "simple_5x5_grouped_3",
    # 'simple_1x1_grouped_3', 
]

upsample = [
    "conv_5x1_1x5",
    "conv_3x1_1x3",
    "simple_3x3",
    "simple_5x5",
    # "growth2_5x5",
    # "growth2_3x3",
    # "decenc_3x3_2",
    # "decenc_5x5_2",
    # "simple_3x3_grouped_3",
    # "simple_5x5_grouped_3",
    # 'simple_1x1_grouped_3',
    # 'simple_1x1', 
 ]

skip = [
    # "decenc_3x3_2",
    # "decenc_5x5_2",
    # "conv_5x1_1x5",
    # "conv_3x1_1x3",
    "simple_1x1",
    "simple_3x3",
    "simple_5x5",
]

# body = [
#     "simple_3x3",
#     "simple_5x5",
#     "simple_3x3_grouped_3",
#     "simple_5x5_grouped_3",
#     "decenc_3x3_2",
#     "decenc_5x5_2",
#     "growth2_5x5",
#     "growth2_3x3",
#     #'simple_1x1_grouped_3',
#     #'simple_1x1', 
# ]

# head = [
#     "simple_3x3",
#     "simple_5x5",
#     "growth2_5x5",
#     "growth2_3x3",
#     "simple_3x3_grouped_3",
#     "simple_5x5_grouped_3",
#     #'simple_1x1_grouped_3',
#     #'simple_1x1', 
# ]

# tail = [
#     "simple_3x3",
#     "simple_5x5",
#     "growth2_5x5",
#     "growth2_3x3",
#     "simple_3x3_grouped_3",
#     "simple_5x5_grouped_3",
#     'simple_1x1_grouped_3',
#     'simple_1x1',
# ]

# upsample = [
#     "conv_5x1_1x5",
#     "conv_3x1_1x3",
#     "simple_3x3",
#     "simple_5x5",
#     # "growth2_5x5",
#     # "growth2_3x3",
#     # "decenc_3x3_2",
#     # "decenc_5x5_2",
#     "simple_3x3_grouped_3",
#     "simple_5x5_grouped_3",
#     # 'simple_1x1_grouped_3',
#     # 'simple_1x1', 
#  ]

# skip = [
#     "decenc_3x3_2",
#     "decenc_5x5_2",
#     "simple_3x3",
#     "simple_5x5",
# ]


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
            C_in, C_out, [bit], C_fixed, 1, affine=False, shared=False
        )
        dag.append(op)
    return nn.Sequential(*dag)


def parse_sr(alpha, name, bits=[2], primitives=None):
    gene = []
    primitives = PRIMITIVES_SR if primitives is None else primitives
    for edges in alpha:
        best_bit = 0
        best_op = 0
        best_val = 0
        n_ops = len(edges) // len(bits)
        for op_idx, edge in enumerate(edges.chunk(n_ops)):
            max_val = edge.max()
            bit_idx = edge.argmax()
            if max_val > best_val:
                best_val = max_val
                best_op = op_idx
                best_bit = bit_idx.item()

        prim = primitives[name][best_op]
        gene.append((prim, bits[best_bit]))
    return gene
