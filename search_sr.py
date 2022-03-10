""" Search cell """
import os
import PIL
import torch
import torch.nn as nn
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
import genotypes as gt
from sr_models.search_cnn import SearchCNNController
from sr_base.datasets import CropDataset
from visualize import plot_sr
import math

from omegaconf import OmegaConf as omg


def train_setup(cfg):

    # INIT FOLDERS & cfg

    cfg.env.save_path = utils.get_run_path(
        cfg.env.log_dir, "SEARCH_" + cfg.env.run_name
    )
    utils.save_scripts(cfg.env.save_path)
    log_handler = utils.LogHandler(cfg.env.save_path + "/log.txt")
    logger = log_handler.create()

    # FIX SEED
    np.random.seed(cfg.env.seed)
    torch.cuda.set_device(cfg.env.gpu)
    np.random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=os.path.join(cfg.env.save_path, "board"))

    writer.add_hparams(
        hparam_dict={str(k): str(cfg[k]) for k in cfg},
        metric_dict={"search/train/loss": 0},
    )

    omg.save(cfg, os.path.join(cfg.env.save_path, "config.yaml"))

    return cfg, writer, logger, log_handler


def run_search(cfg, writer, logger, log_handler):
    # cfg, writer, logger, log_handler = train_setup(cfg)
    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.env.gpu
    torch.cuda.set_device(device)

    train_loader, train_loader_alpha, val_loader = get_data_loaders(cfg)

    model = SearchCNNController(
        cfg.arch.channels,
        cfg.arch.c_fixed,
        cfg.arch.bits,
        cfg.arch.scale,
        cfg.arch.arch_pattern,
        cfg.arch.body_cells,
        device_ids=cfg.env.gpu,
        alpha_selector=cfg.search.alpha_selector,
        quant_noise=cfg.search.get("quant_noise", False),
        skip_mode=cfg.arch.get("skip_mode", True),
        primitives=cfg.arch.get("primitives", None)
    )

    if cfg.search.load_path is not None:
        model.load_state_dict(torch.load(cfg.search.load_path))
        model.eval()
        print(f"loaded a model from: {cfg.search.load_path}")

    base_criterion = nn.L1Loss().to(device)
    criterion = SparseCrit(
        base_criterion,
        cfg.search.epochs,
        type=cfg.search.sparse_type,
        coef=cfg.search.sparse_coef,
    )

    criterion.init_alpha(model.alphas_weights)
    model = model.to(device)

    flops_loss = FlopsLoss(model.n_ops)

    # weights optimizer
    if cfg.search.optimizer == "sgd":
        w_optim = torch.optim.SGD(
            model.weights(),
            cfg.search.w_lr,
            momentum=cfg.search.w_momentum,
            weight_decay=cfg.search.w_weight_decay,
        )
        print("USING SGD")
    else:
        w_optim = torch.optim.Adam(
            model.weights(), 
            cfg.search.w_lr, 
            weight_decay=cfg.search.w_weight_decay,
        )
        print("USING ADAM")

    # alphas optimizer
    alpha_optim = torch.optim.Adam(
        model.alphas_weights(),
        cfg.search.alpha_lr,
        betas=(0.5, 0.999),
        weight_decay=cfg.search.alpha_weight_decay,
    )

    scheduler = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, cfg.search.epochs
        ),
        "linear": torch.optim.lr_scheduler.StepLR(
            w_optim, step_size=3, gamma=0.8
        ),
    }

    lr_scheduler = scheduler[cfg.search.lr_scheduler]

    # training loop
    best_score = 1e3
    cur_step = 0
    temperature = cfg.search.temp_max
    for epoch in range(cfg.search.epochs):
        lr = lr_scheduler.get_last_lr()[0]
        print("LR: ", lr)

        model.print_alphas(logger, cfg.search.temp_max, writer, epoch)

        if epoch >= cfg.search.warm_up:
            temperature = cfg.search.temp_max - (
                cfg.search.temp_max - cfg.search.temp_min
            ) * (epoch - cfg.search.warm_up) / (cfg.search.epochs - 1 - cfg.search.warm_up)

        # training
        score_train, cur_step, best_current_flops = train(
            train_loader,
            train_loader_alpha,
            model,
            criterion,
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
        lr_scheduler.step()
        # validation
        score_val = validate(
            val_loader,
            model,
            criterion,
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
        if best_score > score_val:
            best_score = score_val
            best_flops = best_current_flops
            best_genotype = genotype
            with open(
                os.path.join(cfg.env.save_path, "best_arch.gen"), "w"
            ) as f:
                f.write(str(genotype))
            with open(
                os.path.join(cfg.env.save_path, f"arch_{epoch}.gen"), "w"
            ) as f:
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

            utils.save_checkpoint(model, cfg.env.save_path, is_best)
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
        writer.add_scalar("search/train/temperature", temperature, epoch)

        logger.info("Final best LOSS = {:.3f}".format(best_score))
        logger.info("Best Genotype = {}".format(best_genotype))

    # FINISH TRAINING
    log_handler.close()
    logging.shutdown()
    del model


def log_genotype(
    genotype, cfg, epoch, cur_step, writer, best_current_flops, psnr, best=False
):
    # genotype as an image
    plot_path = os.path.join(
        cfg.env.save_path, cfg.env.im_dir, "EP{:02d}".format(epoch + 1)
    )
    caption = "Epoch {}   FLOPS {:.2e}  search LOSS: {:.3f}".format(
        epoch + 1, best_current_flops, psnr
    )

    im_normal = plot_sr(genotype, plot_path + "-normal", caption)

    im_normal = np.array(
        im_normal.resize(
            (int(im_normal.size[0] / 3), int(im_normal.size[1] // 3)),
            PIL.Image.ANTIALIAS,
        )
    )
    writer.add_image(
        tag=f"SR_im_normal_best_{best}",
        img_tensor=im_normal,
        dataformats="HWC",
        global_step=cur_step,
    )

    writer.add_image(
        tag=f"SR_im_normal_CURRENT",
        img_tensor=im_normal,
        dataformats="HWC",
        global_step=cur_step,
    )


def train(
    train_loader,
    train_alpha_loader,
    model,
    criterion,
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
    loss_meter = utils.AverageMeter()

    stable = True

    writer.add_scalar("search/train/lr", lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y, _, _), (val_X, val_y, _, _)) in enumerate(
        zip(train_loader, train_alpha_loader)
    ):

        trn_X, trn_y = (
            trn_X.to(device, non_blocking=True),
            trn_y.to(device, non_blocking=True),
        )
        val_X, val_y = (
            val_X.to(device, non_blocking=True),
            val_y.to(device, non_blocking=True),
        )
        N = trn_X.size(0)
        if flops_loss.norm == 0:
            model(val_X, stable=True)
            flops_norm, _ = model.fetch_weighted_flops_and_memory()
            flops_loss.set_norm(flops_norm)
            flops_loss.set_penalty(cfg.search.penalty)

        alpha_optim.zero_grad()

        if epoch >= cfg.search.warm_up:
            stable = False

            preds, (flops, mem) = model(val_X, temperature)
            loss = criterion(preds, val_y, epoch) + flops_loss(flops)
            loss.backward()

            if step == len(train_loader) - 1:
                log_weigths_hist(model, writer, epoch, True)
            alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        preds, (flops, mem) = model(trn_X, temperature, stable=False)

        loss_w, init_loss = criterion(preds, trn_y, epoch, get_initial=True)
        loss_w.backward()
    
    
        if step == len(train_loader) - 1:
            log_weigths_hist(model, writer, epoch, False)
            grad_norm(model, writer, epoch)

        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), cfg.search.w_grad_clip)
        w_optim.step()

        loss_meter.update(init_loss.item(), N)

        (
            best_current_flops,
            best_current_memory,
        ) = model.fetch_current_best_flops_and_memory()

        if step % cfg.env.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss: {losses.avg:.3f} Flops: {flops:.2e}".format(
                    epoch + 1,
                    cfg.search.epochs,
                    step,
                    len(train_loader) - 1,
                    losses=loss_meter,
                    flops=best_current_flops,
                )
            )

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
    criterion,
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

            loss = criterion.loss(preds, y)

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
            cfg.env.save_path, x_path[0], y_path[0], preds[0], epoch, writer
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
        cfg.dataset.search_subsample = 0.0001

    split = int(cfg.dataset.search_subsample * n_train * 0.5)

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
            indices[split : split * 2]
        )
        valid_sampler_selection = torch.utils.data.sampler.SubsetRandomSampler(
            indices[split * 2 : split * 3]
            if (split * 3) <= n_train
            else indices[split : split * 2]
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


class SparseCrit(nn.Module):
    def __init__(self, criterion, epochs, type="none", coef=0.1) -> None:
        super().__init__()
        assert type in ["none", "entropy", "l1", "l1_softmax"]
        print("SPARSITY TYPE:", type)
        self.loss = criterion
        self.coef = coef
        self.epochs = epochs
        self.type = type

    def init_alpha(self, alphas):
        self.alphas = alphas

    def forward(self, pred, target, epoch, get_initial=False):
        alpha = self.alphas()
        if self.type == "entropy":
            self.update(epoch)
            loss1 = self.loss(pred, target)
            alpha_prob = [F.softmax(x, dim=-1) for x in alpha]
            ent_loss = torch.sum(
                torch.stack(
                    [torch.sum(torch.mul(i, torch.log(i))) for i in alpha_prob]
                )
            )
            loss2 = -ent_loss
            res = loss1 + self.coef * self.weight1 * self.weight2 * loss2
            return res if not get_initial else (res, loss1)
        elif self.type == "none":
            res = self.loss(pred, target)
            return res if not get_initial else (res, res)
        elif self.type == "l1":
            loss1 = self.loss(pred, target)
            flat_alphas = torch.cat([x.view(-1) for x in alpha])
            l1_regularization = self.coef * torch.norm(flat_alphas, 1)
            res = loss1 + l1_regularization
            return res if not get_initial else (res, loss1)
        elif self.type == "l1_softmax":
            loss1 = self.loss(pred, target)
            flat_alphas = torch.cat([torch.exp(x).view(-1) for x in alpha])
            l1_regularization = self.coef * torch.norm(flat_alphas, 1)
            res = loss1 + l1_regularization
            return res if not get_initial else (res, loss1)

    def update(self, epoch):
        warm_up = self.epochs // 4
        self.weight1 = 1 / (self.epochs - 1) * (epoch)
        self.weight2 = (
            0
            if epoch < warm_up
            else math.log(epoch - warm_up + 2, self.epochs - warm_up + 1)
        )


def log_weigths_hist(model, tb_logger, epoch, log_alpha=False):
    if not log_alpha:
        for name, weight in model.net.named_parameters():
            if "weight" in name:
                if "head" in name or "body" in name:
                    tb_logger.add_histogram(
                        f"weights/{name}", weight.detach().cpu().numpy(), epoch
                    )
                    # tb_logger.add_histogram(
                    #     f"weights_grad/{name}", weight.grad.cpu().numpy(), epoch
                    # )
    else:
        for name in model.alphas:
            for i, alpha in enumerate(model.alphas[name]):
                tb_logger.add_histogram(
                    f"weights_alpha_grad/{name}.{i}", 
                    alpha.grad.cpu().numpy(), 
                    epoch,
                )


def grad_norm(model, tb_logger, epoch):
    norms = {}
    norms["body"] = []
    norms["head"] = []
    norms["tail"] = []
    for name, p in model.named_parameters():
        if ("body" in name) or ("head" in name) or ("tail" in name):
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(1)
                if "body" in name:
                    norms["body"] += [param_norm.item()]

                if "head" in name:
                    norms["head"] += [param_norm.item()]

                if "tail" in name:
                    norms["tail"] += [param_norm.item()]
            else:
                print(f"NONE GRAD in {name}")
    for k in norms:
        norms[k] = np.mean(norms[k]) if norms[k] != [] else 0
    tb_logger.add_scalars(f"search/grad_norms", norms, epoch)

    def grad_per_op(module):
        grad_ops = []
        for op in module._ops:
            grad_op = []
            for p in op.parameters():
                grad_op += [p.grad.detach().data.norm(1).item()]
            grad_op = np.mean(grad_op)
            grad_ops += [grad_op]
        return grad_ops, np.mean([op for op in grad_ops if not op is None])

    net = model.net
    blocks = {
        "head": net.head,
        "upsample": net.upsample,
        "tail": net.tail,
    }
    for i, body in enumerate(net.body):
        blocks[f"body.{i}"] = body.body
        blocks[f"skip.{i}"] = body.skip
    mean_grads = {}
    for name in blocks:
        for i, mixop in enumerate(blocks[name].net):
            mix_grad, mean_grad = grad_per_op(mixop)
            mean_grads[f"{name}.{i}"] = mean_grad
            tb_logger.add_scalars(
                f"grad/{name}.{i}",
                dict(zip(model.primitives[name.split(".")[0]], mix_grad)),
                epoch,
            )
    tb_logger.add_scalars("grads_per_block", mean_grads, epoch)
    return


if __name__ == "__main__":
    CFG_PATH = "./configs/quant_config.yaml"
    cfg = omg.load(CFG_PATH)
    cfg, writer, logger, log_handler = train_setup(cfg)
    run_search(cfg, writer, logger, log_handler)
