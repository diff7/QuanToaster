import os
import random
import warnings
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from models import SRCNN
from datasets import PatchDataset
from utils import AverageMeter, calc_psnr, save_images
from omegaconf import OmegaConf as omg


CONFIG = "./configs/config_50.yaml"


def save_model(cfg, model, psnr):
    if cfg.save_model:
        torch.save(
            model.state_dict(),
            os.path.join(
                cfg.results_dir, "epoch_{}.pth".format(round(psnr, 3))
            ),
        )


def train_one_epoch(
    cfg,
    model,
    train_dataset,
    train_dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    epoch=0,
    logger=None,
):
    model.train()
    epoch_losses = AverageMeter()
    sanity_losses = AverageMeter()

    with tqdm(
        total=(len(train_dataset) - len(train_dataset) % cfg.batch_size)
    ) as t:
        t.set_description("epoch: {}/{}".format(epoch, cfg.num_epochs - 1))

        for i, data in enumerate(train_dataloader):

            inputs, labels, path_input, path_target = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)
            loss_sanity = criterion(inputs, labels)

            epoch_losses.update(loss.item(), len(inputs))
            sanity_losses.update(loss_sanity.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss="{:.6f}".format(epoch_losses.avg))
            t.update(len(inputs))

            if logger is not None:
                if i % cfg.log_step == 0:
                    logger.log({"target - output": epoch_losses.avg})
                    logger.log({"target - input": sanity_losses.avg})

            if i % cfg.save_steps == 0:
                save_images(
                    cfg.results_dir + "/images/",
                    path_input[0],
                    path_target[0],
                    preds[0].detach().unsqueeze(0),
                    i,
                    logger,
                )

    if scheduler is not None:
        scheduler.step()


def eval_and_save(cfg, model, eval_dataloader, device, best_psnr, logger=None):
    model.eval()
    epoch_psnr = AverageMeter()

    for data in eval_dataloader:
        inputs, labels, path_input, path_target = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print("eval psnr: {:.2f}".format(epoch_psnr.avg))
    save_images(
        cfg.results_dir,
        path_input[0],
        path_target[0],
        preds,
        epoch_psnr.avg.item(),
    )

    if logger is not None:
        logger.log({"psnr": epoch_psnr.avg})

    if epoch_psnr.avg > best_psnr:
        best_psnr = epoch_psnr.avg.item()
        save_model(cfg, model, best_psnr)
    return best_psnr


def train_main(cfg):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    writer = SummaryWriter(log_dir=os.path.join(cfg.results_dir, "board"))

    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("using :", device)
    torch.manual_seed(cfg.seed)

    model = SRCNN(
        ker_one=cfg.model.ker_one,
        ker_two=cfg.model.ker_two,
        ker_three=cfg.model.ker_three,
    )

    if not cfg.load_weiths is None:
        print(f"LOADING WEIGHTS: {cfg.load_weiths}")
        state_dict = torch.load(cfg.load_weiths, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)

    criterion = nn.L1oss()

    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {
                "params": model.conv3.parameters(),
                "lr": cfg.lr,
            },  # change last layer LR if required
        ],
        lr=cfg.lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.7
    )

    train_dataset = PatchDataset(cfg, train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataset = PatchDataset(cfg, train=False)
    eval_dataloader = DataLoader(
        dataset=eval_dataset, shuffle=True, batch_size=1
    )

    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(cfg.num_epochs):
        train_one_epoch(
            cfg,
            model,
            train_dataset,
            train_dataloader,
            criterion,
            optimizer,
            device,
            scheduler,
            epoch,
            logger=writer,
        )

        best_psnr = eval_and_save(
            cfg, model, eval_dataloader, device, best_psnr, logger=writer
        )


if __name__ == "__main__":
    cfg = omg.load(CONFIG)
    train_main(cfg)
