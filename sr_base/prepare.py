import os
import imageio
from skimage.util import img_as_ubyte
from tqdm import tqdm
import numpy as np
import PIL.Image as pil_image
from omegaconf import OmegaConf as omg


def save_patch(img, i, j, file, processed_folder, cfg):
    patch = img[i : i + cfg.patch_size, j : j + cfg.patch_size]
    f_name = f'{file.split(".")[0]}_{i}_{j}.png'
    file_path_patch = os.path.join(processed_folder, f_name)
    imageio.imwrite(file_path_patch, img_as_ubyte(patch / 256))


def read_file(file, source_folder):
    file_path = os.path.join(source_folder, file)
    return pil_image.open(file_path).convert("RGB")


def get_files(file, cfg, flag):
    hr = read_file(file, cfg.data_hr + flag)
    file_lr = f'{file.split(".")[0]}x{cfg.scale}.png'
    lr = read_file(file_lr, cfg.data_lr + flag)
    hr, lr = resize_and_convert(cfg, hr, lr)
    return hr, lr


def resize_and_convert(cfg, hr, lr):
    hr_width = (hr.width // cfg.scale) * cfg.scale
    hr_height = (hr.height // cfg.scale) * cfg.scale
    hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    lr = lr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

    hr = np.array(hr).astype(np.float32)
    lr = np.array(lr).astype(np.float32)
    # hr = convert_rgb_to_y(hr)
    # lr = convert_rgb_to_y(lr)
    return hr, lr


def prep_train(cfg):
    flag = cfg.train_folder
    files = [f for f in os.listdir(cfg.data_hr + flag) if "png" in f]
    for file in tqdm(files):

        hr, lr = get_files(file, cfg, flag)
        # used for debug
        # save_file(hr, "hr_" + file, cfg.data)
        # save_file(lr, "lr_" + file, cfg.data)
        for i in range(0, lr.shape[0] - cfg.patch_size + 1, cfg.stride):
            for j in range(0, lr.shape[1] - cfg.patch_size + 1, cfg.stride):
                save_patch(hr, i, j, file, cfg.data_processed_hr + flag, cfg)
                save_patch(lr, i, j, file, cfg.data_processed_lr + flag, cfg)


def save_file(img, file, processed_folder):
    file_path_patch = os.path.join(processed_folder, file)
    imageio.imwrite(file_path_patch, img_as_ubyte(img / 256))


def prep_eval(cfg):
    flag = cfg.val_folder
    files = [f for f in os.listdir(cfg.data_hr + flag) if "png" in f]
    for file in tqdm(files):

        hr, lr = get_files(file, cfg, flag)

        save_file(hr, file, cfg.data_processed_hr + flag)
        save_file(lr, file, cfg.data_processed_lr + flag)


if __name__ == "__main__":

    cfg = omg.load("./configs/config_50.yaml")
    os.makedirs(cfg.data_processed_hr + cfg.train_folder, exist_ok=True)
    os.makedirs(cfg.data_processed_lr + cfg.train_folder, exist_ok=True)

    os.makedirs(cfg.data_processed_hr + cfg.val_folder, exist_ok=True)
    os.makedirs(cfg.data_processed_lr + cfg.val_folder, exist_ok=True)

    print("MAKING TRAIN")
#    prep_train(cfg)
#    print("MAKING VAL")
#    prep_eval(cfg)
