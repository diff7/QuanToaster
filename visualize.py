""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
import genotypes as gt
from PIL import Image


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

    node_n = 0
    for name in parts:
        layers = getattr(genotype, name)
        if name == "body":
            body_start = node_n
            body_end = node_n + len(parts) - 1

        if name == "upsample":
            f = getattr(genotype, "upsample")[0]

            g.edge(
                str(node_n),
                str(node_n + 1),
                label=f"{name.upper()}\n{f[0]}\nbit:{f[1]}+PS",
                fillcolor="darkseagreen2",
            )
            node_n += 1
            pixel_up_node = node_n
            continue

        for op, bit in layers:
            if node_n == 0:
                current_n = "Input"
            else:
                current_n = str(node_n)
            g.edge(
                current_n,
                str(node_n + 1),
                label=f"{name.upper()}\n{op}\nbit:{bit}",
                fillcolor="lightblue",
            )
            node_n += 1

    # body_skip_node
    body_skip = getattr(genotype, "skip")[0]
    g.edge(
        str(body_start),
        str(body_end),
        label=f"skip\n{body_skip[0]}\nbit:{body_skip[1]}",
        fillcolor="gray",
    )
    g.edge(
        str(body_start),
        str(body_end),
        label=f"plain_skip",
        fillcolor="gray",
    )

    # tail skip
    g.edge(
        str(pixel_up_node),
        str(node_n),
        label=f"skip",
        fillcolor="gray",
    )

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
