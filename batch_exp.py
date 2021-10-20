# This script runs search and training using  different hyperparams

import os
import argparse
from search import run_search as run_search_cls
from augment import run_train as run_train_cls

from search_sr import run_search as run_search_sr
from augment_sr import run_train as run_train_sr

from utils import get_run_path
from omegaconf import OmegaConf as omg

from validate_sr import get_model, dataset_loop
import utils

"""
EXAMPLE: python batch_exp.py -t SR -k penalty -v 0.01 0.05 0.1 0.5 0.7 -d gumbel -r 3 -g 3
"""

VAL_CFG_PATH = "./sr_models/valsets4x.yaml"

functions = {
    "SR": [run_search_sr, run_train_sr],
    "CLS": [run_search_cls, run_train_cls],
}


configs = {"SR": "./configs/sr_config.yaml", "CLS": "./configs/config.yaml"}

parser = argparse.ArgumentParser()

parser.add_argument(
    "-k",
    "--key",
    type=str,
    default="penalty",
    help="argument to run different experiments",
)

parser.add_argument(
    "-t",
    "--task",
    type=str,
    default="penalty",
    help="SR or CLS",
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


def run_batch():
    key = args.key
    values = args.values
    base_run_name = args.name

    run_search, run_train = functions[args.task]
    cfg = omg.load(configs[args.task])

    log_dir = cfg.env.log_dir
    assert (key in cfg.train) or (
        key in cfg.search
    ), f"{key} is not found in config"

    for mode in ["train", "search"]:
        cfg[mode].gpu = args.gpu

    for r in range(1, args.repeat + 1):
        for mode in ["train", "search"]:
            cfg.env.log_dir = os.path.join(
                log_dir,
                args.dir,
                mode,
                f"trail_{r}",
            )
            os.makedirs(cfg.env.log_dir, exist_ok=True)

        print("TRIAL #", r)
        for val in values:
            for mode in ["train", "search"]:
                cfg.env.run_name = f"{base_run_name}_{key}_{val}_trail_{r}"

            if key in cfg.search:
                cfg.search[key] = val

            if key in cfg.train:
                cfg.train[key] = val

            # get actual run dir with date stamp

            run_path = get_run_path(
                cfg.env.log_dir, "SEARCH_" + cfg.env.run_name
            )

            cfg.train.genotype_path = os.path.join(
                run_path,
                "best_arch.gen",
            )

            print(f"SEARCHING: {str(key).upper()}:{str(val).upper()}")
            run_search(cfg)
            print(f"TRAINING: {str(key).upper()}:{str(val).upper()}")
            run_train(cfg)

            with open(cfg.train.genotype_path, "r") as f:
                genotype = utils.from_str(f.read())

            weights_path = os.path.join(cfg.train.save, "best.pth.tar")

            # VALIDATE:
            logger = utils.get_logger(run_path + "/validation_log.txt")
            save_dir = os.path.join(run_path, "FINAL_VAL")
            os.makedirs(save_dir, exist_ok=True)
            logger.info(genotype)
            valid_cfg = omg.load(VAL_CFG_PATH)
            model = get_model(
                weights_path, cfg.train.gpu, genotype, blocks=cfg.train.blocks
            )
            dataset_loop(valid_cfg, model, logger, save_dir, cfg.train.gpu)


if __name__ == "__main__":
    run_batch()
