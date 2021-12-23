import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random

def check_image_file(filename):
    r"""Filter non image files in directory.
    Args:
        filename (str): File name under path.
    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    return any(
        filename.endswith(extension)
        for extension in [
            "bmp",
            ".png",
            ".jpg",
            ".jpeg",
            ".png",
            ".PNG",
            ".jpeg",
            ".JPEG",
        ]
    )


class ValidationSet(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, cfg):
        r"""
        Args:
            input_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
        """
        super(ValidationSet, self).__init__()

        folder_path_hr = os.path.join(cfg.data_processed_hr)
        folder_path_lr = os.path.join(cfg.data_processed_lr)

        lr_files = os.listdir(folder_path_lr)
        hr_files = os.listdir(folder_path_hr)

        self.input_filenames = [
            os.path.join(folder_path_lr, x)
            for x in lr_files
            if check_image_file(x)
        ]
        self.target_filenames = [
            os.path.join(folder_path_hr, x)
            for x in hr_files
            if check_image_file(x)
        ]

        print(f"DATASET SIZE:{len(self.input_filenames)}")

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )  # Note - to tensor divides by 255

    def __getitem__(self, index):
        r"""Get image source file.
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image, high resolution image.
        """

        input_img = Image.open(self.input_filenames[index]).convert("RGB")
        target_img = Image.open(self.target_filenames[index]).convert("RGB")

        input_img = self.transforms(input_img)
        target_img = self.transforms(target_img)

        return (
            input_img,
            target_img,
            self.input_filenames[index],
            self.target_filenames[index],
        )

    def __len__(self):
        return len(self.input_filenames)


class CropDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, cfg, train=True):

        super(CropDataset, self).__init__()

        self.crop_size = cfg.crop_size
        self.scale = cfg.scale
        self.train = train

        if self.train:
            path = cfg.files_list + "/train.list"
        else:
            path = cfg.files_list + "/val.list"

        with open(path, "r") as f:
            files = f.read()
            self.files_paths = files.split("\n")
            self.input_filenames = [s.split("/")[-1] for s in self.files_paths]

        self.input_filenames = [
            x for x in self.input_filenames if check_image_file(x)
        ]

        print(f"DATASET SIZE:{len(self.input_filenames)}")

        self.transforms_hr = transforms.Compose(
            [transforms.ToTensor()]  # Note - to tensor divides by 255
        )

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]  # Note - to tensor divides by 255
        )

    def random_crop_image(self, image):
        width, height = image.size
        crop_size = self.crop_size * self.scale
        x_start = random.randint(0, width - crop_size)
        y_start = random.randint(0, height - crop_size)

        return image.crop(
            (x_start, y_start, x_start + crop_size, y_start + crop_size)
        )

    def random_flip(self, img):
        if random.random() > 0.5:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        return img

    def downscale(self, image):
        if self.train:
            size_lr = self.crop_size
            return image, image.resize((size_lr, size_lr), Image.BICUBIC)
        else:
            width, height = image.size
            width, height = width // self.scale, height // self.scale
            return (
                image.resize(
                    (width * self.scale, height * self.scale), Image.BICUBIC
                ),
                image.resize((width, height), Image.BICUBIC),
            )

    def __getitem__(self, index):
        r"""Get image source file.
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image, high resolution image.
        """

        hr = Image.open(self.files_paths[index]).convert("RGB")
        if self.train == True:
            hr = self.random_crop_image(hr)
            hr = self.random_flip(hr)

        hr, lr = self.downscale(hr)
        hr_img = self.transforms(hr)
        lr_img = self.transforms(lr)
        return (
            lr_img,
            hr_img,
            self.files_paths[index],
            self.files_paths[index],
        )

    def __len__(self):
        return len(self.input_filenames)
