# This script runs search and training using  different hyperparams

import os
import argparse

from search_sr import run_search
from augment_sr import run_train

from search_sr import train_setup as search_setup
from augment_sr import train_setup as augment_setup

from omegaconf import OmegaConf as omg

from validate_sr import get_model, dataset_loop
import genotypes
import utils
import traceback


"""
EXAMPLE: 
python batch_exp.py -k penalty -v 0.0 0.001 0.005 -d l1 -r 1 -g 1
"""

VAL_CFG_PATH = "./sr_models/valsets4x.yaml"

config = "./configs/sr_config.yaml"

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
    "-n", "--name", type=str, default="batch_experiment", help="experiment name"
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

    cfg = omg.load(config)

    log_dir = cfg.env.log_dir
    assert (key in cfg.train) or (
        key in cfg.search
    ), f"{key} is not found in config"

    cfg.env.gpu = args.gpu

    for r in range(1, args.repeat + 1):
        for mode in ["train", "search"]:
            cfg.env.log_dir = os.path.join(
                log_dir,
                args.dir,
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

            run_path = utils.get_run_path(
                cfg.env.log_dir, "SEARCH_" + cfg.env.run_name
            )

            cfg.train.genotype_path = os.path.join(
                run_path,
                "best_arch.gen",
            )

            print(f"SEARCHING: {str(key).upper()}:{str(val).upper()}")
            cfg_search, writer, logger, log_handler = search_setup(cfg)
            try:
                run_search(cfg_search, writer, logger, log_handler)
            except Exception as e:
                with open(os.path.join(cfg_search.env.save_path, "ERROR.txt"), "a") as f:
                    f.write(traceback.format_exc())
                    print(traceback.format_exc())
                raise e


            print(f"TRAINING: {str(key).upper()}:{str(val).upper()}")
            cfg_train, writer, logger, log_handler = augment_setup(cfg)
            try:
                run_train(cfg_train, writer, logger, log_handler)
            except Exception as e:
                with open(os.path.join(cfg_train.env.save_path, "ERROR.txt"), "a") as f:
                    f.write(traceback.format_exc())
                    print(traceback.format_exc())
                raise e
            



            with open(cfg.train.genotype_path, "r") as f:
                genotype = genotypes.from_str(f.read())

            weights_path = os.path.join(cfg.env.save_path, "best.pth.tar")

            # VALIDATE:
            logger = utils.get_logger(run_path + "/validation_log.txt")
            save_dir = os.path.join(run_path, "FINAL_VAL")
            os.makedirs(save_dir, exist_ok=True)
            logger.info(genotype)
            valid_cfg = omg.load(VAL_CFG_PATH)

            model = get_model(
                weights_path,
                cfg.env.gpu,
                genotype,
                cfg.arch.c_fixed,
                cfg.arch.channels,
                cfg.dataset.scale,
                body_cells=cfg.arch.body_cells,
            )
            dataset_loop(valid_cfg, model, logger, save_dir, cfg.env.gpu)


if __name__ == "__main__":
    run_batch()
