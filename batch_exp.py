# This script runs search and training using  different hyperparams

import os
import argparse
from search import run_search
from augment import run_train
from utils import get_run_path
from omegaconf import OmegaConf as omg

"""
EXAMPLE: python batch_exp.py -k penalty -v 0.01 0.05 0.1 0.5 0.7 -d gumbel_plus -r 3 -g 3
"""
parser = argparse.ArgumentParser()

parser.add_argument(
    "-k",
    "--key",
    type=str,
    default="penalty",
    help="argument to run different experiments",
)

parser.add_argument(
    "-v",
    "--values",
    nargs="+",
    default=[],
    help="argument values ex.: 0.1 0.2 0.3 ",
)

parser.add_argument("-d", "--dir", type=str, default="batch", help="log dir")

parser.add_argument(
    "-n", "--name", type=str, default="batch experiment", help="experiment name"
)

parser.add_argument(
    "-r",
    "--repeat",
    type=int,
    default=1,
    help="repeat experiments",
)

parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu to use")

args = parser.parse_args()


def run_batch(CFG_PATH):
    cfg = omg.load(CFG_PATH)
    key = args.key
    values = args.values
    base_run_name = args.name
    log_dir = {"search": cfg.search.log_dir, "train": cfg.search.log_dir}

    assert (key in cfg.train) or (
        key in cfg.search
    ), f"{key} is not found in config"

    for mode in ["train", "search"]:
        cfg[mode].gpu = args.gpu

    for r in range(1, args.repeat + 1):
        for mode in ["train", "search"]:
            cfg[mode].log_dir = os.path.join(
                log_dir[mode],
                args.dir,
                f"trail_{r}",
            )
            os.makedirs(cfg[mode].log_dir, exist_ok=True)

        print("TRIAL #", r)
        for val in values:
            for mode in ["train", "search"]:
                cfg[mode].run_name = f"{base_run_name}_{key}_{val}"

            if key in cfg.search:
                cfg.search[key] = val

            if key in cfg.train:
                cfg.train[key] = val

            # get actual run dir with date stamp

            run_path = get_run_path(
                cfg["search"].log_dir, "SEARCH_" + cfg["search"].run_name
            )

            cfg.train.genotype_path = os.path.join(
                run_path,
                "best_arch.gen",
            )

            print(f"SEARCHING: {str(key).upper()}:{str(val).upper()}")
            run_search(cfg)
            print(f"TRAINING: {str(key).upper()}:{str(val).upper()}")
            run_train(cfg)


if __name__ == "__main__":
    CFG_PATH = "./configs/config.yaml"
    run_batch(CFG_PATH)
