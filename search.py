""" Search cell """
import os
from types import new_class
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot

from omegaconf import OmegaConf as omg

CFG_PATH = "./configs/config.yaml"


def train_setup():

    # INIT FOLDERS & cfg

    cfg = omg.load(CFG_PATH).search
    cfg.save = utils.get_run_path(cfg.log_dir, "SEARCH_" + cfg.run_name)

    logger = utils.get_logger(cfg.save + "/log.txt")

    # FIX SEED
    np.random.seed(cfg.seed)
    torch.cuda.set_device(cfg.gpu)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.save, "board"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"search/train/loss": 0},
    )

    return cfg, writer, logger


def main():
    cfg, writer, logger = train_setup()
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.gpu
    torch.cuda.set_device(device)

    # set seed
    loaders, input_channels, n_classes = get_data_loaders(cfg)
    train_loader, train_loader_alpha, val_loader = loaders
    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(
        input_channels,
        cfg.init_channels,
        n_classes,
        cfg.layers,
        net_crit,
        device_ids=cfg.gpu,
    )
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(
        model.weights(),
        cfg.w_lr,
        momentum=cfg.w_momentum,
        weight_decay=cfg.w_weight_decay,
    )
    # alphas optimizer
    alpha_optim = torch.optim.Adam(
        model.alphas(),
        cfg.alpha_lr,
        betas=(0.5, 0.999),
        weight_decay=cfg.alpha_weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, cfg.epochs, eta_min=cfg.w_lr_min
    )
    architect = Architect(model, cfg.w_momentum, cfg.w_weight_decay)

    # training loop
    best_top1 = 0.0
    cur_step = 0
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        top1_train, top5_train, cur_step = train(
            train_loader,
            train_loader_alpha,
            model,
            architect,
            w_optim,
            alpha_optim,
            lr,
            epoch,
            writer,
            logger,
            cfg,
            device,
            cur_step,
        )

        # validation
        top1_val, top5_val = validate(
            val_loader, model, epoch, cur_step, logger, cfg, device
        )

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        log_genotype(genotype, cfg, epoch, cur_step, writer, best=False)

        # save
        if best_top1 < top1_val:
            best_top1 = top1_val
            best_genotype = genotype
            with open(os.path.join(cfg.save, "best_arch.gen"), "w") as f:
                f.write(str(genotype))

            is_best = True
        else:
            is_best = False

        log_genotype(best_genotype, cfg, epoch, cur_step, writer, best=True)
        utils.save_checkpoint(model, cfg.save, is_best)
        print("")

        writer.add_scalars(
            "top1/search", {"val": top1_val, "train": top1_train}, epoch
        )

        logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
        logger.info("Best Genotype = {}".format(best_genotype))


def log_genotype(genotype, cfg, epoch, cur_step, writer, best=False):
    # genotype as a image
    plot_path = os.path.join(cfg.save, cfg.im_dir, "EP{:02d}".format(epoch + 1))
    caption = "Epoch {}".format(epoch + 1)

    im_normal = plot(genotype.normal, plot_path + "-normal", caption)
    im_reduce = plot(genotype.reduce, plot_path + "-reduce", caption)

    writer.add_image(
        tag=f"im_normal_best_{best}",
        img_tensor=np.array(im_normal),
        dataformats="HWC",
        global_step=cur_step,
    )
    writer.add_image(
        tag=f"im_reduce_best_{best}",
        img_tensor=np.array(im_reduce),
        dataformats="HWC",
        global_step=cur_step,
    )


def train(
    train_loader,
    train_alpha_loader,
    model,
    architect,
    w_optim,
    alpha_optim,
    lr,
    epoch,
    writer,
    logger,
    cfg,
    device,
    cur_step,
):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    writer.add_scalar("search/train/lr", lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(
        zip(train_loader, train_alpha_loader)
    ):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(
            device, non_blocking=True
        )
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(
            device, non_blocking=True
        )
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), cfg.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=losses,
                    top1=top1,
                    top5=top5,
                )
            )

        (
            weighted_flops,
            weighted_memory,
        ) = model.fetch_weighted_flops_and_memory()
        writer.add_scalar("search/train/loss", loss.item(), cur_step)
        writer.add_scalar(
            "search/train/weighted_flops", weighted_flops.item(), cur_step
        )
        writer.add_scalar(
            "search/train/weighted_memory", weighted_memory.item(), cur_step
        )

        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(
            epoch + 1, cfg.epochs, top1.avg
        )
    )

    return top1.avg, top5.avg, cur_step


def validate(valid_loader, model, epoch, cur_step, logger, cfg, device):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1,
                        cfg.epochs,
                        step,
                        len(valid_loader) - 1,
                        losses=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

    logger.info(
        "Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(
            epoch + 1, cfg.epochs, top1.avg
        )
    )

    return top1.avg, top5.avg


def get_data_loaders(cfg):
    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        cfg.dataset, cfg.data_path, cutout_length=0, validation=False
    )

    # split data to train/validation
    n_train = len(train_data)
    if cfg.debug_mode:
        cfg.train_portion = 0.001

    split = int(np.floor(cfg.train_portion * n_train))
    leftover = int(np.floor((1 - cfg.train_portion) * n_train)) // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        indices[:split]
    )
    if cfg.debug_mode:
        train_sampler_alpha = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]
        )
        valid_sampler_selection = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]
        )
    else:
        train_sampler_alpha = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split : split + leftover]
        )
        valid_sampler_selection = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split + leftover : split + leftover * 2]
        )

    loaders = []
    for sampler in [
        train_sampler,
        train_sampler_alpha,
        valid_sampler_selection,
    ]:
        print("SET SIZE:", len(sampler))
        loaders.append(
            torch.utils.data.DataLoader(
                train_data,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=cfg.workers,
                pin_memory=True,
            )
        )

    return loaders, input_channels, n_classes


if __name__ == "__main__":
    main()
