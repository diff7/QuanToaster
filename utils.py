""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc
import time

from PIL import Image
from genotypes import from_str


def get_run_path(base_dir, run_name):
    run_dir = "{}-{}".format(run_name, time.strftime("%Y-%m-%d-%H"))
    run_dir = os.path.join(base_dir, run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_sr_data(cfg):
    pass


def get_data(dataset, data_path, cutout_length, validation):
    """Get torchvision dataset"""
    dataset = dataset.lower()

    if dataset == "cifar10":
        dset_cls = dset.CIFAR10
        n_classes = 10
        input_size = 32
        input_channels = 3
    elif dataset == "mnist":
        dset_cls = dset.MNIST
        n_classes = 10
        input_size = 28
        input_channels = 1
    elif dataset == "fashionmnist":
        dset_cls = dset.FashionMNIST
        n_classes = 10
        input_size = 28
        input_channels = 1
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(
        dataset, cutout_length
    )
    trn_data = dset_cls(
        root=data_path, train=True, download=True, transform=trn_transform
    )

    # assuming shape is NHW or NHWC

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(
            dset_cls(
                root=data_path,
                train=False,
                download=True,
                transform=val_transform,
            )
        )

    return ret


def get_logger(file_path):
    """Make python logger"""
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger(file_path.split('/')[-1])
    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """Compute parameter size in MB"""
    n_params = sum(
        np.prod(v.size())
        for k, v in model.named_parameters()
        if not k.startswith("aux_head")
    )
    return n_params / 1024.0 / 1024.0


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, "best.pth.tar")
        shutil.copyfile(filename, best_filename)


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


def min_max(m):
    mx = m.max()
    mn = m.min()
    return (m - m.min()) / (mx - mn)


def prepare_images(path_input, path_target, out):
    out = out.permute(1, 2, 0)
    out = min_max(out)
    input_img = Image.open(path_input)
    target_img = Image.open(path_target)

    out_image = out.mul(255.0).cpu().numpy()

    out_image = np.clip(out_image, 0.0, 255.0).astype(np.uint8)
    out_image = Image.fromarray(out_image)

    return target_img, input_img, out_image


def save_images(
    results_dir, path_input, path_target, out, cur_iter, logger=None
):

    cur_iter = round(cur_iter, 3)
    results_dir = os.path.join(results_dir, "images")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    target, input_img, out_image = prepare_images(path_input, path_target, out)

    if logger is not None:
        logger.add_image(
            tag=f"target",
            img_tensor=np.array(target),
            dataformats="HWC",
            global_step=cur_iter,
        )
        logger.add_image(
            tag=f"input_img",
            img_tensor=np.array(input_img),
            dataformats="HWC",
            global_step=cur_iter,
        )
        logger.add_image(
            tag=f"out_image",
            img_tensor=np.array(out_image),
            dataformats="HWC",
            global_step=cur_iter,
        )

    target.save(f"{results_dir}/taret_step_{cur_iter}.png")
    input_img.save(f"{results_dir}/input_step_{cur_iter}.png")
    out_image.save(f"{results_dir}/out_image_step_{cur_iter}.png")
