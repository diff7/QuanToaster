import os
import torch
import numpy as np
from PIL import Image


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
