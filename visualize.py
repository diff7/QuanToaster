""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
import genotypes as gt
from PIL import Image

"""
DARTS EXAMPLE:

Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
                [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], 
                [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], 
                [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]], 
                normal_concat=range(2, 6), 

                reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
                [('max_pool_3x3', 0), ('skip_connect', 2)], 
                [('max_pool_3x3', 0), ('skip_connect', 2)], 
                [('max_pool_3x3', 0), ('skip_connect', 2)]], 
                reduce_concat=range(2, 6))
  

"""


def plot(genotype, file_path, caption=None):
    """make DAG plot and save to file_path as .png"""
    edge_attr = {"fontsize": "20", "fontname": "times"}
    node_attr = {
        "style": "filled",
        "shape": "rect",
        "align": "center",
        "fontsize": "20",
        "height": "0.5",
        "width": "0.5",
        "penwidth": "2",
        "fontname": "times",
    }
    g = Digraph(
        format="png", edge_attr=edge_attr, node_attr=node_attr, engine="dot"
    )
    g.body.extend(["rankdir=LR"])

    # input nodes
    g.node("c_{k-2}", fillcolor="darkseagreen2")
    g.node("c_{k-1}", fillcolor="darkseagreen2")

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor="lightblue")

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("CONCAT", fillcolor="palegoldenrod")
    for i in range(n_nodes):
        g.edge(str(i), "Out", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap="false", fontsize="20", fontname="times")

    g.render(file_path, view=False)
    im = Image.open(file_path + ".png").convert("RGB")
    return im


"""
SR EXAMPLE:

Genotype(normal=[[('simple_1x1_grouped_full', 0)], 
                [('simple_1x1_grouped_full', 0), ('simple_1x1_grouped_full', 1)], 
                [('simple_1x1_grouped_full', 0), ('simple_1x1', 1)], 
                [('growth4_3x3_grouped_full', 3), ('simple_1x1_grouped_full', 2)]], 
                normal_concat=range(4, 6))        
"""


def plot_sr(genotype, file_path, caption=None):
    """make DAG plot and save to file_path as .png"""
    edge_attr = {"fontsize": "20", "fontname": "times"}
    node_attr = {
        "style": "filled",
        "shape": "rect",
        "align": "center",
        "fontsize": "20",
        "height": "0.5",
        "width": "0.5",
        "penwidth": "2",
        "fontname": "times",
    }
    g = Digraph(
        format="png", edge_attr=edge_attr, node_attr=node_attr, engine="dot"
    )
    g.body.extend(["rankdir=LR"])

    # input nodes
    g.node("Input", fillcolor="darkseagreen2")

    parts = ["head", "body", "upsample", "tail"]

    node_n = 1
    for name in parts:
        layers = getattr(genotype, name)
        if name == "body":
            body_start = node_n
            body_end = node_n + len(parts)

        if name == "upsample":
            g.edge(
                current_n,
                str(node_n + 1),
                label=f"{name}_PIXEL_SHUFFLE",
                fillcolor="gray",
            )
            node_n += 1

        for op in layers:
            if node_n == 0:
                current_n = "Input"
            current_n = str(node_n)
            g.edge(
                current_n,
                str(node_n + 1),
                label=f"{name}_{op}",
                fillcolor="gray",
            )
            node_n += 1

    # BODY SKIP
    body_SKIP
    # SKIP NODE
    g.node("Pixel shuffle", fillcolor="palegoldenrod")
    g.edge(
        "Input",
        "Pixel shuffle",
        label=str(genotype[-1][0][0]),
        fillcolor="gray",
    )
    # output node

    # for i in range(n_nodes - 1, n_nodes):
    g.edge(str(i + 1), "Pixel shuffle", label="", fillcolor="gray")

    # g.edge("Input", "Pixel shuffle", label="sum +", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap="false", fontsize="20", fontname="times")

    g.render(file_path, view=False)
    im = Image.open(file_path + ".png").convert("RGB")
    return im


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} GENOTYPE".format(sys.argv[0]))

    genotype_path = sys.argv[1]

    with open(genotype_path, "r") as f:
        genotype_str = gt.from_str(f.read())
        print(genotype_str)
    try:
        genotype = genotype_str  # gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    if type(genotype) == "Genotype":
        plot(genotype.normal, "./examples/normal")
        plot(genotype.reduce, "./examples/reduction")
    else:
        plot_sr(genotype.normal, "./examples/normal")
