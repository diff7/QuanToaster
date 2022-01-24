""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
import genotypes as gt
from PIL import Image


def plot_sr(genotype, file_path, caption=None):
    """make DAG plot and save to file_path as .png"""
    edge_attr = {"fontsize": "20", "fontname": "helvetica", "penwidth": "1.5", "arrowsize": "1.0", "arrowhead": "vee"}
    node_attr = {
        "style": "filled",
        "shape": "doublecircle",
        "align": "center",
        "fontsize": "20",
        "height": "0.5",
        "width": "0.5",
        "penwidth": "2",
        "fontname": "helvetica",
        "color": "black",
        "fillcolor": "gold1"
    }
    g = Digraph(
        format="png", edge_attr=edge_attr, node_attr=node_attr, engine="dot"
    )
    g.body.extend(["rankdir=LR"])

    # input nodes
    g.node("Input", fillcolor="chartreuse1")

    parts = ["head", "body", "upsample", "tail"]

    node_n = 0
    for name in parts:
        layers = getattr(genotype, name)
        if name == "body":
            body_start = node_n + 1
            body_end = node_n + len(layers) + 1 
            g.node(
                str(node_n),
                fillcolor="gray",
                width="0.6",
                fixedsize="True",
                shape="triangle",
                orientation="90",
                fontsize="18"
            )
            node_n += 1
            g.edge(
                str(node_n - 1),
                str(node_n),
                arrowhead="none"
            )

        if name == "upsample":
            f = getattr(genotype, "upsample")[0]
            g.node(
                str(node_n + 1),
                fillcolor="deepskyblue",
                fixedsize="True",
                width="1.0"
            )
            g.edge(
                str(node_n),
                str(node_n + 1),
                label=f"{name.upper()}\n{f}+PS",
                fillcolor="darkseagreen2",
            )
            node_n += 1
            pixel_up_node = node_n
            continue

        for op in layers:
            if node_n == 0:
                current_n = "Input"
            else:
                current_n = str(node_n)
            g.edge(
                current_n,
                str(node_n + 1),
                label=f"{name.upper()}\n{op}",
                fillcolor="lightblue",
            )
            node_n += 1
        if name == "body":
            g.node(
                str(node_n + 1),
                fillcolor="gray",
                width="0.6",
                fixedsize="True",
                shape="triangle",
                orientation="270",
                fontsize="18"
            )
            node_n += 1
            g.edge(
                str(node_n - 1),
                str(node_n),
                arrowhead="none"
            )

    # body_skip_node
    body_skip = getattr(genotype, "skip")[0]
    g.edge(
        str(body_start),
        str(body_end),
        label=f"skip\n{body_skip}",
        fillcolor="gray",
    )
    g.edge(
        str(body_start),
        str(body_end),
        label=f"plain_skip",
        fillcolor="gray",
        style="dashed"
    )

    g.edge(
        str(body_start - 1),
        str(body_end + 1),
        label=f"skip",
        fillcolor="gray",
        style="dashed"
    )
    # tail skip
    g.edge(
        str(pixel_up_node),
        str(node_n),
        label=f"skip",
        fillcolor="gray",
        style="dashed"
    )

    # layers = getattr(genotype, "body")
    # for i, op in enumerate(layers):
    #     g.edge(
    #         f"b.{i}",
    #         f"b.{i + 1}",
    #         label=f"BODY\n{op}",
    #         width="1000.",
            
    #     )

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
        plot_sr(genotype, "./examples/sr_arch")