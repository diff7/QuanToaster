""" Search cell """
import os
from types import new_class
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
from sr_models.search_cnn import SearchCNNController
from architect import Architect, ArchConstrains
from sr_base.datasets import PatchDataset
from visualize import plot_sr

from omegaconf import OmegaConf as omg


def train_setup(cfg):

    # INIT FOLDERS & cfg

    cfg_dataset = cfg.dataset
    cfg = cfg.search
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

    with open(os.path.join(cfg.save, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{str(k)}:{str(v)}\n")

    return cfg, writer, logger, cfg_dataset


def run_search(cfg):
    cfg, writer, logger, cfg_dataset = train_setup(cfg)
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.gpu
    torch.cuda.set_device(device)

    train_loader, train_loader_alpha, val_loader = get_data_loaders(cfg_dataset)
    criterion = nn.L1Loss().to(device)

    model = SearchCNNController(
        cfg.channels,
        cfg.repeat_factor,
        criterion,
        cfg.n_nodes,
        device_ids=cfg.gpu,
        use_soft_edge=cfg.use_soft_edge,
        alpha_selector=cfg.alpha_selector,
    )

    if cfg.use_adjuster:
        ConstrainAdjuster = ArchConstrains(**cfg.adjuster, device=device)
    model = model.to(device)

    flops_loss = FlopsLoss(model.n_ops)

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

    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, cfg.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            w_optim, step_size=3, gamma=0.8
        ),
    }

    lr_scheduler = scheduler[cfg.lr_scheduler]

    architect = Architect(model, cfg.w_momentum, cfg.w_weight_decay)

    # training loop
    best_score = -1e3
    cur_step = 0
    temperature = cfg.temperature_start
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger, cfg.temperature_start)
        model.print_edges(logger)

        if epoch > cfg.warm_up:
            temperature *= cfg.temp_red

        # training
        score_train, cur_step, best_current_flops = train(
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
            flops_loss,
            temperature,
        )

        if cfg.use_adjuster:
            ConstrainAdjuster.adjust(model)

        # validation
        score_val = validate(
            val_loader,
            model,
            epoch,
            logger,
            writer,
            cfg,
            device,
            best=False,
            temperature=temperature,
        )

        score_val_unsummed = validate(
            val_loader,
            model,
            epoch,
            logger,
            writer,
            cfg,
            device,
            best=True,
            temperature=temperature,
        )

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        log_genotype(
            genotype,
            cfg,
            epoch,
            cur_step,
            writer,
            best_current_flops,
            best=False,
        )

        # save
        if best_score < score_val:
            best_score = score_val
            best_flops = best_current_flops
            best_genotype = genotype
            with open(os.path.join(cfg.save, "best_arch.gen"), "w") as f:
                f.write(str(genotype))

            writer.add_scalar("search/best_val", best_score, epoch)
            writer.add_scalar("search/best_flops", best_flops, epoch)

            is_best = True

            log_genotype(
                best_genotype,
                cfg,
                epoch,
                cur_step,
                writer,
                best_flops,
                best=True,
            )

            utils.save_checkpoint(model, cfg.save, is_best)
            print("")

        writer.add_scalars(
            "psnr/search", {"val": best_score, "train": score_train}, epoch
        )
        writer.add_scalars(
            "psnr_val_unsummed/search", {"val": score_val_unsummed}, epoch
        )
        writer.add_scalar("search/train/temperature", temperature, epoch)

        logger.info("Final best Prec@1 = {:.3f}".format(best_score))
        logger.info("Best Genotype = {}".format(best_genotype))


def log_genotype(
    genotype, cfg, epoch, cur_step, writer, best_current_flops, best=False
):
    # genotype as a image
    plot_path = os.path.join(cfg.save, cfg.im_dir, "EP{:02d}".format(epoch + 1))
    caption = "Epoch {}   FLOPS {:.02E}".format(epoch + 1, best_current_flops)

    im_normal = plot_sr(genotype.normal, plot_path + "-normal", caption)

    writer.add_image(
        tag=f"SR_im_normal_best_{best}",
        img_tensor=np.array(im_normal),
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
    flops_loss,
    temperature,
):
    psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    stable = True

    writer.add_scalar("search/train/lr", lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y, _, _), (val_X, val_y, _, _)) in enumerate(
        zip(train_loader, train_alpha_loader)
    ):

        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(
            device, non_blocking=True
        )
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(
            device, non_blocking=True
        )
        N = trn_X.size(0)
        if flops_loss.norm == 0:
            model(val_X)
            flops_norm, _ = model.fetch_weighted_flops_and_memory()
            flops_loss.set_norm(flops_norm)
            flops_loss.set_penalty(cfg.penalty)

        # phase 2. architect step (alpha)

        alpha_optim.zero_grad()

        if epoch >= cfg.warm_up:
            stable = False

            if cfg.unrolled:
                architect.backward(
                    trn_X, trn_y, val_X, val_y, lr, w_optim, flops_loss
                )
            else:

                preds, (flops, mem) = model(val_X, temperature)
                if cfg.use_l1_alpha:
                    alphas = model.alphas()
                    flat_alphas = torch.cat([x.view(-1) for x in alphas])
                    l1_regularization = cfg.l1_lambda * torch.norm(
                        flat_alphas, 1
                    )
                    loss = (
                        model.criterion(preds, val_y)
                        + flops_loss(flops)
                        + l1_regularization
                    )
                else:
                    loss = model.criterion(preds, val_y) + flops_loss(flops)
                loss.backward()
                alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        preds, (flops, mem) = model(trn_X, temperature, stable=stable)

        loss_w = model.criterion(preds, trn_y)
        loss_w.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), cfg.w_grad_clip)
        w_optim.step()

        psnr = utils.calc_psnr(preds, trn_y)
        loss_meter.update(loss_w.item(), N)
        psnr_meter.update(psnr.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} "
                "PSNR ({psnr.avg:.3f}) ".format(
                    epoch + 1,
                    cfg.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    psnr=psnr_meter,
                )
            )

        (
            best_current_flops,
            best_current_memory,
        ) = model.fetch_current_best_flops_and_memory()
        writer.add_scalar("search/train/loss", loss_w, cur_step)

        writer.add_scalar(
            "search/train/best_current_flops", best_current_flops, cur_step
        )

        writer.add_scalar(
            "search/train/best_current_memory", best_current_memory, cur_step
        )

        writer.add_scalar("search/train/flops_loss", flops, cur_step)
        writer.add_scalar("search/train/weighted_flops", flops.item(), cur_step)
        writer.add_scalar("search/train/weighted_memory", mem.item(), cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final PSNR {:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )

    return psnr_meter.avg, cur_step, best_current_flops


def validate(
    valid_loader,
    model,
    epoch,
    logger,
    writer,
    cfg,
    device,
    best=False,
    temperature=1,
):
    psnr_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y, x_path, y_path) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )

            N = X.size(0)
            if best:
                preds = model.forward_current_best(X)
            else:
                preds, (flops, mem) = model(X, temperature)

            loss = model.criterion(preds, y)

            psnr = utils.calc_psnr(preds, y)
            loss_meter.update(loss.item(), N)
            psnr_meter.update(psnr.item(), N)
            if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} "
                    "PSNR ({psnr.avg:.3f}) ".format(
                        epoch + 1,
                        cfg.epochs,
                        step,
                        len(valid_loader) - 1,
                        losses=loss_meter,
                        psnr=psnr_meter,
                    )
                )

    logger.info(
        "Valid: [{:2d}/{}] Final Prec@1 {:.3f}".format(
            epoch + 1, cfg.epochs, psnr_meter.avg
        )
    )
    if not best:
        utils.save_images(
            cfg.save, x_path[0], y_path[0], preds[0], epoch, writer
        )
    return psnr_meter.avg


def get_data_loaders(cfg):

    # get data with meta info
    train_data = PatchDataset(cfg, train=True)

    # split data to train/validation
    n_train = len(train_data)
    if cfg.debug_mode:
        cfg.train_portion = 0.02

    split = int(np.floor(cfg.train_portion * n_train))
    leftover = int(np.floor((1 - cfg.train_portion) * n_train)) // 2
    indices = list(range(n_train))
    random.shuffle(indices)

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
                pin_memory=False,
            )
        )
    return loaders


class FlopsLoss:
    def __init__(self, n_ops, reduce=4):
        self.n_ops = n_ops / reduce
        self.norm = 0

    def set_norm(self, norm):
        self.norm = norm.detach() * self.n_ops
        self.min = norm.detach() / self.n_ops

    def set_penalty(self, penalty):
        self.penalty = float(penalty)

    def __call__(self, weighted_flops):
        l = (weighted_flops - self.min) / (self.norm - self.min)
        return l * self.penalty


if __name__ == "__main__":
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    run_search(cfg)
