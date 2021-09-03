import os
import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image


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


def fitler_subset(orig, sub):
    return set(orig).intersection(set(sub))


class PatchDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, cfg, train=True):
        r"""
        Args:
            input_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
        """
        super(PatchDataset, self).__init__()
        if train:
            folder_type = "train/"
        else:
            folder_type = "valid/"

        folder_path_hr = os.path.join(cfg.data_processed_hr, folder_type)
        folder_path_lr = os.path.join(cfg.data_processed_lr, folder_type)

        lr_files = os.listdir(folder_path_lr)
        hr_files = os.listdir(folder_path_hr)

        if (cfg.subset is not None) and train:
            print("FILTERING A SUBSET")
            with open(cfg.subset, "r") as f:
                subset = f.read()
                subset = subset.split("\n")

            subset = [s.split("/")[-1] for s in subset]
            hr_files = fitler_subset(hr_files, subset)
            lr_files = fitler_subset(lr_files, subset)

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
            [transforms.ToTensor()]  # Note - to tensor divides by 255
        )

    def __getitem__(self, index):
        r"""Get image source file.
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image, high resolution image.
        """

        input_img = Image.open(self.input_filenames[index])
        target_img = Image.open(self.target_filenames[index])

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
