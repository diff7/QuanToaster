import os
import argparse

from augment_sr import run_train

from augment_sr import train_setup 

from omegaconf import OmegaConf as omg

from validate_sr import get_model, dataset_loop
import genotypes
import utils
import traceback


BLOCK_START = 0
BLOCK_STOP = 5

if __name__ == "__main__":
    VAL_CFG_PATH = "./sr_models/valsets4x.yaml"
    CFG_PATH = "./configs/sr_config.yaml"

    for blocks in range(BLOCK_START, BLOCK_STOP):
        cfg = omg.load(CFG_PATH)
        cfg.env.run_name = f"{blocks}"
        cfg.env.log_dir = "/home/dev/data_main/LOGS/SR/LongBicubic/"
        cfg.train.blocks = blocks
        cfg.train.genotype_path = "./genotype_longsrcnn.gen"

        print(f"TRAINING: {blocks} blocks")

        cfg, writer, logger, log_handler = train_setup(cfg)
        try:
            run_train(cfg, writer, logger, log_handler, arch_type="LongSRCNN")
        except Exception as e:
            with open(os.path.join(cfg.env.save_path, "ERROR.txt"), "a") as f:
                f.write(traceback.format_exc())
                print(traceback.format_exc())
            raise e

        # VALIDATE:
        genotype = f"{blocks}_blocks"
        weights_path = os.path.join(cfg.env.save_path, "best.pth.tar")
        logger = utils.get_logger(cfg.env.save_path + "/validation_log.txt")
        save_dir = os.path.join(cfg.env.save_path, "FINAL_VAL")
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
            body_cells=blocks,
            arch_type="LongSRCNN"

        )
        dataset_loop(valid_cfg, model, logger, save_dir, cfg.env.gpu)