import os
import random
import genotypes
import genotypes as gt
from validate_sr import get_model, dataset_loop
from augment_sr import run_train

from omegaconf import OmegaConf as omg


VAL_CFG_PATH = "./sr_models/valsets4x.yaml"


def get_r_in_list(l):
    idx = random.randint(0, len(l) - 1)
    return l[idx]


def make_random_genotype(cfg):
    gen = dict()

    for name in gt.PRIMITIVES_SR:
        blocks = []
        for _ in range(cfg.arch.arch_pattern[name]):
            ops = get_r_in_list(gt.PRIMITIVES_SR[name])
            blocks.append((ops, 32))

        gen[name] = blocks

    return gt.Genotype_SR(**gen)


def sample_random(cfg):

    for r in range(100):
        cfg.env.log_dir = os.path.join(
            "random_sampling",
            f"trail_{r}",
        )
        os.makedirs(cfg.env.log_dir, exist_ok=True)

        run_train(cfg)
        with open(cfg.train.genotype_path, "r") as f:
            genotype = genotypes.from_str(f.read())

        weights_path = os.path.join(cfg.env.save, "best.pth.tar")

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
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    print(make_random_genotype(cfg))