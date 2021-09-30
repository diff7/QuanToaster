""" Training augmented model """
import os
import torch
import torch.nn as nn
import random
import logging
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf as omg
from sr_models.test_arch import ManualCNN, ESPCN

from sr_models.augment_cnn import AugmentCNN
import utils
from sr_base.datasets import CropDataset, PatchDataset
from genotypes import from_str


def train_setup(cfg):

    # INIT FOLDERS & cfg
    cfg_dataset = copy.copy(cfg.dataset)
    repeat_factor = cfg.search.repeat_factor
    channels = cfg.search.channels
    cfg = cfg.train

    cfg.channels = channels
    cfg.repeat_factor = repeat_factor

    cfg.save = utils.get_run_path(cfg.log_dir, "TUNE_" + cfg.run_name)

    logger = utils.get_logger(cfg.save + "/log.txt")

    # FIX SEED
    np.random.seed(cfg.seed)
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.save, "board_train"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"tune/train/loss": 0},
    )

    with open(os.path.join(cfg.save, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{str(k)}:{str(v)}\n")

    return cfg, writer, logger, cfg_dataset


def run_train(cfg):
    cfg, writer, logger, cfg_dataset = train_setup(cfg)
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.gpu
    torch.cuda.set_device(device)
    cfg_dataset.subset = None
    train_data = PatchDataset(cfg_dataset, train=True)
    val_data = PatchDataset(cfg_dataset, train=False)

    if cfg_dataset.debug_mode:
        indices = list(range(300))
        random.shuffle(indices)
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:150]
        )
    else:
        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(
            list(range(len(train_data)))
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        sampler=sampler_train,
        # shuffle=True,
        num_workers=cfg.workers,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        # sampler=sampler_val,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=False,
    )

    criterion = nn.L1Loss().to(device)

    with open(cfg.genotype_path, "r") as f:
        genotype = from_str(f.read())

    writer.add_text(tag="tune/arch/", text_string=str(genotype))
    print(genotype)

    ## model = ManualCNN(cfg.channels, cfg.repeat_factor)
    model = ESPCN(4)
    # model = AugmentCNN(
    #     cfg.channels,
    #     cfg.repeat_factor,
    #     genotype,
    # )

    model.to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    writer.add_text(
        tag="ModelParams",
        text_string=str("Model size = {:.3f} MB".format(mb_params)),
    )

    # weights optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        cfg.lr,
        ##momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.7
        ),
    }

    lr_scheduler = scheduler[cfg.lr_scheduler]

    best_score = 0.0
    # training loop
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        if cfg.use_drop_prob:
            drop_prob = cfg.drop_path_prob * (1 - epoch / cfg.epochs)
            print("DROP PATH", drop_prob)
            model.drop_path_prob(drop_prob)

        # training
        score_train = train(
            train_loader,
            model,
            optimizer,
            criterion,
            epoch,
            writer,
            logger,
            device,
            cfg,
        )

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        score_val = validate(
            val_loader,
            model,
            criterion,
            epoch,
            cur_step,
            writer,
            logger,
            device,
            cfg,
        )

        # save
        if best_score < score_val:
            score_val = score_val
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, cfg.save, is_best)

        print("")
        writer.add_scalars(
            "psnr/tune", {"val": score_val, "train": score_train}, epoch
        )

    logger.info("Final best PSNR = {:.4%}".format(best_score))

    # FINISH TRAINING
    logging.shutdown()
    del model


def train(
    train_loader,
    model,
    optimizer,
    criterion,
    epoch,
    writer,
    logger,
    device,
    cfg,
):
    psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar("tune/train/lr", cur_lr, cur_step)

    model.train()

    for step, (X, y, _, _) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        psnr = utils.calc_psnr(preds, y)
        loss_meter.update(loss.item(), N)
        psnr_meter.update(psnr.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "PSNR ({score.avg:.3f})".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    score=psnr_meter,
                )
            )

        writer.add_scalar("tune/train/loss", loss_meter.avg, cur_step)
        writer.add_scalar("tune/train/psnr", psnr_meter.avg, cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:3d}/{}] Final PSNR{:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )
    return psnr_meter.avg


def validate(
    valid_loader, model, criterion, epoch, cur_step, writer, logger, device, cfg
):
    psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (
            X,
            y,
            x_path,
            y_path,
        ) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )
            N = X.size(0)

            preds = model(X).clamp(0.0, 1.0)
            loss = criterion(preds, y)

            psnr = utils.calc_psnr(preds, y)
            loss_meter.update(loss.item(), N)
            psnr_meter.update(psnr.item(), N)

        if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
            logger.info(
                "VAL: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "PSNR ({score.avg:.3f})".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(valid_loader) - 1,
                    losses=loss_meter,
                    score=psnr_meter,
                )
            )

    writer.add_scalar("tune/val/loss", loss_meter.avg, cur_step)
    writer.add_scalar("tune/val/psnr", psnr_meter.avg, cur_step)

    logger.info(
        "Valid: [{:3d}/{}] Final PSNR{:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )

    indx = random.randint(0, len(x_path) - 1)
    utils.save_images(
        cfg.save, x_path[indx], y_path[indx], preds[indx], epoch, writer
    )

    return psnr_meter.avg


if __name__ == "__main__":
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    run_train(cfg)