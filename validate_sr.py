import os
import torch
import random
import logging
from omegaconf import OmegaConf as omg
from genotypes import from_str
from sr_models.augment_cnn import AugmentCNN
import utils
from sr_base.datasets import ValidationSet
from genotypes import from_str
import pandas as pd


def get_model(
    weights_path,
    device,
    genotype,
    c_fixed=32,
    channels=3,
    scale=4,
    body_cells=4,
):
    model = AugmentCNN(
        channels,
        c_fixed,
        scale,
        genotype,
        blocks=body_cells,
    )

    model_ = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(model_)
    model.to(device)
    return model


def run_val(model, cfg_val, save_dir, device):
    # set default gpu device id
    torch.cuda.set_device(device)
    val_data = ValidationSet(cfg_val)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        # sampler=sampler_val,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    score_val = validate(val_loader, model, device, save_dir)
    random_image = torch.randn(1, 3, 32, 32).cuda(device)
    _ = model(random_image)
    flops_32, _ = model.fetch_info()

    random_image = torch.randn(1, 3, 256, 256).cuda(device)
    _ = model(random_image)
    flops_256, _ = model.fetch_info()

    mb_params = utils.param_size(model)
    return score_val, flops_32, flops_256, mb_params


def validate(valid_loader, model, device, save_dir):
    psnr_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for (
            X,
            y,
            x_path,
            y_path,
        ) in valid_loader:
            X, y = X.to(device, non_blocking=True), y.to(device)
            N = X.size(0)

            preds = model(X).clamp(0.0, 1.0)

            psnr = utils.compute_psnr(preds, y)
            psnr_meter.update(psnr, N)

    indx = random.randint(0, len(x_path) - 1)
    utils.save_images(
        save_dir,
        x_path[indx],
        y_path[indx],
        preds[indx],
        cur_iter=0,
        logger=None,
    )

    return psnr_meter.avg


def dataset_loop(valid_cfg, model, logger, save_dir, device):
    df = pd.DataFrame(columns=["Model size", "Flops(32x32)", "Flops(256x256)", "PSNR"])
    for dataset in valid_cfg:
        os.makedirs(os.path.join(save_dir, str(dataset)), exist_ok=True)
        score_val, flops_32, flops_256, mb_params = run_val(
            model,
            valid_cfg[dataset],
            os.path.join(save_dir, str(dataset)),
            device,
        )
        logger.info("\n{}:".format(str(dataset)))
        logger.info("Model size = {:.3f} MB".format(mb_params))
        logger.info("Flops = {:.2e} operations 32x32".format(flops_32))
        logger.info("Flops = {:.2e} operations 256x256".format(flops_256))
        logger.info("PSNR = {:.3f}%".format(score_val))
        df.loc[str(dataset)] = [mb_params, flops_32, flops_256, score_val]
    df.to_csv(os.path.join(save_dir, "..", "validation_df.csv"))


if __name__ == "__main__":
    CFG_PATH = "./sr_models/valsets4x.yaml"
    valid_cfg = omg.load(CFG_PATH)
    run_name = "TEST_2"
    genotype_path = "/home/dev/data_main/LOGS/QUANT/bilevel_search_v2/trail_1/SEARCH_batch experiment_penalty_0.1_trail_1-2021-12-18-15/best_arch.gen"
    weights_path = "/home/dev/data_main/LOGS/QUANT/bilevel_search_v2/trail_1/TUNE_batch experiment_penalty_0.1_trail_1-2021-12-18-21/best.pth.tar"
    log_dir = "/home/dev/data_main/LOGS/QUANT/"
    save_dir = os.path.join(log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    channels = 3
    repeat_factor = 16
    device = 1

    with open(genotype_path, "r") as f:
        genotype = from_str(f.read())

    logger = utils.get_logger(save_dir + "/validation_log.txt")
    logger.info(genotype)

    model = get_model(
        weights_path,
        device,
        genotype,
        c_fixed=36,
        channels=3,
        scale=4,
        body_cells=3,
    )
    dataset_loop(valid_cfg, model, logger, save_dir, device)
