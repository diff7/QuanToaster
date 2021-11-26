""" Search cell """
import os
import torch
import torch.nn as nn
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
from sr_models.search_cnn import SearchCNNController
from architect import Architect, ArchConstrains
from sr_base.datasets import CropDataset
from visualize import plot_sr

from omegaconf import OmegaConf as omg


def train_setup(cfg):

    # INIT FOLDERS & cfg

    cfg.env.save = utils.get_run_path(
        cfg.env.log_dir, "SEARCH_" + cfg.env.run_name
    )

    log_handler = utils.LogHandler(cfg.env.save + "/log.txt")
    logger = log_handler.create()

    # FIX SEED
    np.random.seed(cfg.env.seed)
    torch.cuda.set_device(cfg.env.gpu)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.env.save, "board"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"search/train/loss": 0},
    )

    with open(os.path.join(cfg.env.save, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{str(k)}:{str(v)}\n")

    return cfg, writer, logger, log_handler


def run_search(cfg):
    cfg, writer, logger, log_handler = train_setup(cfg)
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.env.gpu
    torch.cuda.set_device(device)

    train_loader, train_loader_alpha, val_loader = get_data_loaders(cfg)
    criterion = nn.L1Loss().to(device)

    model = SearchCNNController(
        cfg.arch.channels,
        cfg.arch.c_fixed,
        cfg.arch.bits,
        cfg.arch.scale,
        criterion,
        cfg.arch.arch_pattern,
        cfg.arch.body_cells,
        device_ids=cfg.env.gpu,
        alpha_selector=cfg.search.alpha_selector,
    )

    if cfg.search.load_path is not None:
        model = torch.load(cfg.search.load_path)
        print(f"loaded a model from: {cfg.search.load_path}")

    if cfg.search.use_adjuster:
        ConstrainAdjuster = ArchConstrains(**cfg.search.adjuster, device=device)
    model = model.to(device)

    flops_loss = utils.FlopsLoss(model.n_ops)
    FlopsReg = utils.FlopsScheduler(
        start_reg=cfg.search.reg_sched.start_reg,
        reg_step=cfg.search.reg_sched.reg_step,
        start_after=cfg.search.reg_sched.start_after,
        step=cfg.search.reg_sched.step,
        max_reg=cfg.search.reg_sched.max_reg,
    )

    # weights optimizer
    w_optim = torch.optim.SGD(
        model.weights(),
        cfg.search.w_lr,
        momentum=cfg.search.w_momentum,
        weight_decay=cfg.search.w_weight_decay,
    )
    # alphas optimizer
    alpha_optim = torch.optim.Adam(
        model.alphas_weights(),
        cfg.search.alpha_lr,
        betas=(0.5, 0.999),
        weight_decay=cfg.search.alpha_weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler()

    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, cfg.search.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            w_optim, step_size=3, gamma=0.8
        ),
    }

    lr_scheduler = scheduler[cfg.search.lr_scheduler]

    architect = Architect(
        model, cfg.search.w_momentum, cfg.search.w_weight_decay
    )

    # training loop
    best_score = 1e3
    cur_step = 0
    temperature = cfg.search.temperature_start
    for epoch in range(cfg.search.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger, cfg.search.temperature_start)

        if FlopsReg(epoch) > 0:
            temperature *= cfg.search.temp_red

        # training
        score_train, cur_step, best_current_flops = train(
            train_loader,
            train_loader_alpha,
            model,
            architect,
            w_optim,
            alpha_optim,
            scaler,
            lr,
            epoch,
            writer,
            logger,
            cfg,
            device,
            cur_step,
            flops_loss,
            temperature,
            FlopsReg,
        )

        if cfg.search.use_adjuster:
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

        # log genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # save
        if FlopsReg.register:
            with open(
                os.path.join(
                    cfg.env.save, f"arch_ep_{epoch}_reg_{FlopsReg(epoch)}.gen"
                ),
                "w",
            ) as f:
                f.write(str(genotype))

        if best_score > score_val:
            best_score = score_val
            best_flops = best_current_flops
            best_genotype = genotype
            with open(os.path.join(cfg.env.save, "best_arch.gen"), "w") as f:
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
                best_score,
                best=True,
            )

            utils.save_checkpoint(model, cfg.env.save, is_best)
            print("")
        else:
            log_genotype(
                genotype,
                cfg,
                epoch,
                cur_step,
                writer,
                best_current_flops,
                score_val,
                best=False,
            )
        print("best current", best_current_flops)
        writer.add_scalars(
            "loss/search", {"val": best_score, "train": score_train}, epoch
        )
        writer.add_scalar(
            "search/train/reg_scheduler", FlopsReg(epoch), cur_step
        )

        writer.add_scalar("search/train/temperature", temperature, epoch)

        logger.info("Final best LOSS = {:.3f}".format(best_score))
        logger.info("Best Genotype = {}".format(best_genotype))
        log_weigths_hist(model, writer, epoch)

    # FINISH TRAINING
    log_handler.close()
    logging.shutdown()
    del model


def log_genotype(
    genotype, cfg, epoch, cur_step, writer, best_current_flops, psnr, best=False
):
    # genotype as a image
    plot_path = os.path.join(
        cfg.env.save, cfg.env.im_dir, "EP{:02d}".format(epoch + 1)
    )
    caption = "Epoch {}   FLOPS {:.2e}  search LOSS: {:.3f}".format(
        epoch + 1, best_current_flops, psnr
    )

    im_normal = plot_sr(genotype, plot_path + "-normal", caption)

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
    scaler,
    lr,
    epoch,
    writer,
    logger,
    cfg,
    device,
    cur_step,
    flops_loss,
    temperature,
    FlopsReg,
):
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
            flops_loss.set_penalty(cfg.search.penalty)

        # phase 2. architect step (alpha)

        alpha_optim.zero_grad()

        if FlopsReg(epoch) > 0:
            stable = False

            preds, (flops, mem) = model(val_X, temperature)
            if cfg.search.use_l1_alpha:
                alphas = model.alphas()
                flat_alphas = torch.cat([x.view(-1) for x in alphas])
                l1_regularization = cfg.search.l1_lambda * torch.norm(
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
        with torch.cuda.amp.autocast():
            preds, (flops, mem) = model(trn_X, temperature, stable=stable)
            loss_w = model.criterion(preds, trn_y)
        scaler.scale(loss_w).backward()

        # gradient clipping
        # nn.utils.clip_grad_norm_(model.weights(), cfg.search.w_grad_clip)

        scaler.step(w_optim)
        scaler.update()
        flops_loss.set_penalty(FlopsReg(epoch))
        loss_meter.update(loss_w.item(), N)

        (
            best_current_flops,
            best_current_memory,
        ) = model.fetch_current_best_flops_and_memory()

        if step % cfg.env.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} BitOPS: {flops:.4e}".format(
                    epoch + 1,
                    cfg.search.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    flops=best_current_flops,
                )
            )
        # print("NORMS: ", grad_norm(model))
        writer.add_scalar("search/train/loss", loss_w, cur_step)
        writer.add_scalar(
            "search/train/best_current_flops", best_current_flops, cur_step
        )

        writer.add_scalar(
            "search/train/best_current_memory", best_current_memory, cur_step
        )

        writer.add_scalar("search/train/flops_loss", flops, cur_step)
        writer.add_scalar("search/train/weighted_flops", flops, cur_step)
        writer.add_scalar("search/train/weighted_memory", mem, cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final LOSS {:.3f}".format(
            epoch + 1, cfg.search.epochs, loss_meter.avg
        )
    )

    return loss_meter.avg, cur_step, best_current_flops


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

            loss_meter.update(loss.item(), N)
            if step % cfg.env.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} ".format(
                        epoch + 1,
                        cfg.search.epochs,
                        step,
                        len(valid_loader) - 1,
                        losses=loss_meter,
                    )
                )

    logger.info(
        "Valid: [{:2d}/{}] Final LOSS {:.3f}".format(
            epoch + 1, cfg.search.epochs, loss_meter.avg
        )
    )
    if not best:
        utils.save_images(
            cfg.env.save, x_path[0], y_path[0], preds[0], epoch, writer
        )
    return loss_meter.avg


def get_data_loaders(cfg):

    # get data with meta info
    train_data = CropDataset(cfg.dataset, train=True)

    # split data to train/validation
    n_train = len(train_data)
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    if cfg.dataset.debug_mode:
        cfg.dataset.train_portion = 0.001

    split = int(np.floor(cfg.dataset.train_portion * n_train))
    leftover = int(
        np.floor(
            (1 - cfg.dataset.train_portion) * n_train * cfg.dataset.val_portion
        )
    )

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        indices[:split]
    )
    if cfg.dataset.debug_mode:
        train_sampler_alpha = torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]
        )
        valid_sampler_selection = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split : split * 2]
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
                batch_size=cfg.dataset.batch_size,
                sampler=sampler,
                num_workers=cfg.env.workers,
                pin_memory=False,
            )
        )
    return loaders


def log_weigths_hist(model, tb_logger, epoch):
    for name, weight in model.named_parameters():
        if "weight" in name:
            if "head" in name or "body" in name:
                tb_logger.add_histogram(name, weight, epoch)
                tb_logger.add_histogram(f"{name}.grad", weight.grad, epoch)


# def grad_norm(model):
#     norms = {}
#     norms["body"] = 0
#     norms["head"] = 0
#     norms["tail"] = 0
#     for name, p in model.named_parameters():
#         if ("body" in name) or ("head" in name) or ("tail" in name):
#             if p.grad is not None:
#                 param_norm = p.grad.detach().data.norm(2)
#                 if "body" in name:
#                     norms["body"] += param_norm.item() ** 2

#                 if "head" in name:
#                     norms["head"] += param_norm.item() ** 2
#                     print("head", param_norm.item())

#                 if "tail" in name:
#                     norms["tail"] += param_norm.item() ** 2
#                     print("tail", param_norm.item())
#     return norms


if __name__ == "__main__":
    CFG_PATH = "./configs/sr_config.yaml"
    cfg = omg.load(CFG_PATH)
    run_search(cfg)
