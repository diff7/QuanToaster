import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.totensor = transforms.ToTensor()

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            lr = self.totensor(f["lr"][idx])
            hr = self.totensor(f["hr"][idx])

            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["lr"])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.totensor = transforms.ToTensor()

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            lr = self.totensor(f["lr"][str(idx)][:, :])
            hr = self.totensor(f["hr"][str(idx)][:, :])
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["lr"])
