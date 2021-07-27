""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf as omg

from models.augment_cnn import AugmentCNN
import utils
from genotypes import from_str

CFG_PATH = "./configs/config.yaml"


def train_setup():

    # INIT FOLDERS & cfg

    cfg = omg.load(CFG_PATH).train
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

    return cfg, writer, logger


def main():
    cfg, writer, logger = train_setup()

    logger.info("Logger is set - training start")

    # set default gpu device id
    device = cfg.gpu
    torch.cuda.set_device(device)

    # get data with meta info
    (
        input_size,
        input_channels,
        n_classes,
        train_data,
        valid_data,
    ) = utils.get_data(
        cfg.dataset, cfg.data_path, cfg.cutout_length, validation=True
    )

    criterion = nn.CrossEntropyLoss().to(device)

    use_aux = cfg.aux_weight > 0.0

    with open(cfg.genotype_path, "r") as f:
        genotype = from_str(f.read())
    print(genotype)
    model = AugmentCNN(
        input_size,
        input_channels,
        cfg.init_channels,
        n_classes,
        cfg.layers,
        use_aux,
        genotype,
    )

    model.to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs
    )

    best_top1 = 0.0
    # training loop
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        drop_prob = cfg.drop_path_prob * epoch / cfg.epochs
        model.drop_path_prob(drop_prob)

        # training
        top1_train = train(
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
        top1_val = validate(
            valid_loader,
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
        if best_top1 < top1_val:
            best_top1 = top1_val
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, cfg.save, is_best)

        print("")
        writer.add_scalars(
            "top1/tune", {"val": top1_val, "train": top1_train}, epoch
        )

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


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
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar("tune/train/lr", cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if cfg.aux_weight > 0.0:
            loss += cfg.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
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

        writer.add_scalar("tune/train/loss", loss.item(), cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(
            epoch + 1, cfg.epochs, top1.avg
        )
    )
    return top1.avg


def validate(
    valid_loader, model, criterion, epoch, cur_step, writer, logger, device, cfg
):
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

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
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

    writer.add_scalar("tune/val/loss", losses.avg, cur_step)

    logger.info(
        "Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(
            epoch + 1, cfg.epochs, top1.avg
        )
    )

    return top1.avg


if __name__ == "__main__":
    main()
