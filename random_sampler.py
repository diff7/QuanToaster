import os
import random
import argparse
import genotypes as gt
from validate_sr import get_model, dataset_loop
from augment_sr import run_train

from omegaconf import OmegaConf as omg
import utils


VAL_CFG_PATH = "./sr_models/valsets4x.yaml"

"""
EXAMPLE: python3 random_sampler.py -d random -g 1 -r 1
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default="batch", help="log dir")
parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu to use")
parser.add_argument("-r", "--runs", type=int, default=10, help="gpu to use")
parser.add_argument("-s", "--start", type=int, default=0, help="gpu to use")
args = parser.parse_args()


def get_r_in_list(l):
    idx = random.randint(0, len(l) - 1)
    return l[idx]


def make_random_genotype(cfg):
    gen = dict()

    for name in gt.PRIMITIVES_SR:
        blocks = []
        for _ in range(cfg.arch.arch_pattern[name]):
            ops = get_r_in_list(gt.PRIMITIVES_SR[name])
            bits = get_r_in_list(cfg.arch.bits)
            blocks.append((ops, bits))

        gen[name] = blocks

    return gen


def set_fp(gen):
    fp_gen = dict()
    for name in gen:
        blocks = []
        for ops, bits in gen[name]:
            blocks.append((ops, 32))
        fp_gen[name] = blocks

    return fp_gen


def train_loop(cfg):
    cfg.env.gpu = args.gpu
    log_dir = cfg.env.log_dir

    for r in range(args.start, args.runs):
        cfg.env.log_dir = os.path.join(log_dir, args.dir)
        os.makedirs(cfg.env.log_dir, exist_ok=True)

        random_gen = make_random_genotype(cfg)
        random_fp_gen = set_fp(random_gen)
        tp = ["fp", "quant"]

        for i, genotype in enumerate([random_fp_gen, random_gen]):
            genotype = gt.Genotype_SR(**genotype)
            print(genotype)

            cfg.env.run_name = f"{args.dir}_trail_{r}_{tp[i]}"

            run_path = utils.get_run_path(cfg.env.log_dir, cfg.env.run_name)
            gen_path = os.path.join(run_path, "genotype.gen")
            cfg.train.genotype_path = gen_path

            with open(gen_path, "w") as f:
                f.write(str(genotype))

            run_train(cfg)

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
                skip_mode=cfg.arch.skip_mode,
            )

            dataset_loop(valid_cfg, model, logger, save_dir, cfg.env.gpu)


if __name__ == "__main__":
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    train_loop(cfg)
