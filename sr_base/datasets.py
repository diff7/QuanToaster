import os
import random
import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image


from sr_base.imgproc import center_crop
from sr_base.imgproc import image2tensor
from sr_base.imgproc import random_crop
from sr_base.imgproc import random_horizontally_flip
from sr_base.imgproc import random_rotate


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
        elif train == False:
            folder_type = "valid/"
        else:
            folder_type = ""

        folder_path_hr = os.path.join(cfg.data_processed_hr, folder_type)
        folder_path_lr = os.path.join(cfg.data_processed_lr, folder_type)

        lr_files = os.listdir(folder_path_lr)
        hr_files = os.listdir(folder_path_hr)

        if "subset" in cfg:
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
            [transforms.ToTensor()]
        )  # Note - to tensor divides by 255

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

    def crop_image(self, image):
        width, height = image.size
        crop_size = self.crop_size * self.scale
        x_start = random.randint(0, width - crop_size)
        y_start = random.randint(0, height - crop_size)

        return image.crop(
            (x_start, y_start, x_start + crop_size, y_start + crop_size)
        )

    def downscale(self, image):
        if self.train:
            size_lr = self.crop_size
            return image, image.resize((size_lr, size_lr))
        else:
            width, height = image.size
            width, height = width // self.scale, height // self.scale
            return (
                image.resize((width * self.scale, height * self.scale)),
                image.resize((width, height)),
            )

    def __getitem__(self, index):
        r"""Get image source file.
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image, high resolution image.
        """

        hr = Image.open(self.files_paths[index])
        if self.train == True:
            hr = self.crop_image(hr)

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


class AugmentLoader(torch.utils.data.dataset.Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.
    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, cfg, train=True) -> None:
        # Get the index of all images in the high-resolution folder and low-resolution folder
        # under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        super(AugmentLoader, self).__init__()

        if train:
            folder_type = "train/"
        elif train == False:
            folder_type = "valid/"
        else:
            folder_type = ""

        self.mode = train

        folder_path_hr = os.path.join(cfg.data_processed_hr, folder_type)
        folder_path_lr = os.path.join(cfg.data_processed_lr, folder_type)

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

        self.image_size = cfg.crop_size  # HR image size.
        self.upscale_factor = cfg.scale

    def __getitem__(self, index):
        lr = Image.open(self.input_filenames[index])
        hr = Image.open(self.target_filenames[index])

        # Data enhancement methods.
        if self.mode == "train":
            lr, hr = random_crop(lr, hr, self.image_size, self.upscale_factor)
            lr, hr = random_rotate(lr, hr, 90)
            lr, hr = random_horizontally_flip(lr, hr, 0.5)
        else:
            lr, hr = center_crop(lr, hr, self.image_size, self.upscale_factor)

        # `PIL.Image` image data is converted to `Tensor` format data.
        lr = image2tensor(lr)
        hr = image2tensor(hr)

        return (
            lr,
            hr,
            self.input_filenames[index],
            self.target_filenames[index],
        )

    def __len__(self) -> int:
        return len(self.filenames)