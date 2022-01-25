""" Utilities """
import os
import math
import logging
import shutil
import torch
import numpy as np
import time

import shutil
from PIL import Image


def get_run_path(base_dir, run_name):
    run_dir = "{}-{}".format(run_name, time.strftime("%Y-%m-%d-%H"))
    run_dir = os.path.join(base_dir, run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_scripts(run_path):
    dest = os.path.join(run_path, "code_copy/")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree("./", dest)


class LogHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def create(self):
        logger = logging.getLogger(self.file_path.split("/")[-1])
        log_format = "%(asctime)s | %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
        self.file_handler = logging.FileHandler(self.file_path)
        self.file_handler.setFormatter(formatter)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(formatter)

        logger.addHandler(self.file_handler)
        logger.addHandler(self.stream_handler)
        logger.setLevel(logging.INFO)

        self.logger = logger
        return self.logger

    def close(self):
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.stream_handler)


def get_logger(file_path):
    """Make python logger"""
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger(file_path.split("/")[-1])
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


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


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


def save_checkpoint(model, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, "best.pth.tar")
        shutil.copyfile(filename, best_filename)


def compute_psnr(img1, img2):
    img1 = tensor2img_np(img1)
    img2 = tensor2img_np(img2)
    img1 = rgb2y(img1[4:-4, 4:-4, :])
    img2 = rgb2y(img2[4:-4, 4:-4, :])
    return psnr(img1, img2)


def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def tensor2img_np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze(0)
    tensor = tensor.float().cpu().clamp_(*min_max)  # Clamp is for on hard_tanh
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But receieved tensor with dimension = %d"
            % n_dim
        )
    if out_type == np.uint8:
        img_np = (
            img_np * 255.0
        ).round()  # This is important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def rgb2y(img):
    assert img.dtype == np.uint8
    in_img_type = img.dtype
    img.astype(np.float64)
    img_y = (
        (np.dot(img[..., :3], [65.481, 128.553, 24.966])) / 255.0 + 16.0
    ).round()
    return img_y.astype(in_img_type)


def min_max(m):
    mx = m.max()
    mn = m.min()
    return (m - m.min()) / (mx - mn)


def prepare_images(path_input, path_target, out):
    out = out.permute(1, 2, 0)
    out = min_max(out)
    if path_input is not None:
        input_img = Image.open(path_input)
    else:
        input_img = None
    if path_target is not None:
        target_img = Image.open(path_target)
    else:
        path_target = None

    out_image = out.mul(255.0).cpu().numpy()

    out_image = np.clip(out_image, 0.0, 255.0).astype(np.uint8)
    if out_image.shape[-1] == 1:
        out_image = out_image.squeeze(-1)
        out_image = np.stack([out_image, out_image, out_image], axis=2)
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
        if not target is None:
            logger.add_image(
                tag=f"target",
                img_tensor=np.array(target),
                dataformats="HWC",
                global_step=cur_iter,
            )
        if not input_img is None:
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


class FlopsScheduler:
    def __init__(
        self, start_reg=0, start_after=0, reg_step=0, step=1, max_reg=1e10
    ):
        self.start_after = start_after
        self.cur_reg = start_reg
        self.step = step
        self.reg_step = reg_step
        self.max_reg = max_reg
        self.cur_epoch = start_after
        self.register = False

    def __call__(self, epoch):
        if epoch > self.cur_epoch:
            self.cur_epoch = epoch + self.step
            self.set_reg()
        if self.cur_epoch - 1 == epoch:
            self.register = True
        else:
            self.register = False
        return self.cur_reg

    def set_reg(self):
        if self.cur_reg < self.max_reg:
            self.cur_reg += self.reg_step


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
