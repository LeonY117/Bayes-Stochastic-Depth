import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from typing import Dict, Optional, Tuple
from torch import Tensor

from tqdm import tqdm

from .memory_dataset import MemoryDataset

__all__ = ["load_CIFAR"]


class _CustomTransforms(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # self.normalize = transforms.Normalize(0.5, 0.5)

    def __call__(self, x):
        # x, y = sample
        x = self.to_tensor(x)
        x = x.to(torch.float)
        x = self.normalize(x)
        return x


def _train_val_split(train_set, train_ratio=0.9, deterministic=True):
    """
    Split the training set into training and validation set
    """
    if deterministic:
        torch.manual_seed(0)
    train_size = int(train_ratio * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_size, val_size]
    )
    return train_set, val_set


def _move_dataset_to_device(
    dataset: Dataset, device: "str" = "cuda:0"
) -> Tuple[Tensor, Tensor]:
    """
    Takes a dataset object and loads everything onto the gpu
    """
    # placeholder tensors
    c, H, W = dataset[0][0].shape
    n = len(dataset)
    X = torch.zeros((n, c, H, W), dtype=torch.float, device=device, requires_grad=False)
    Y = torch.zeros((n), dtype=torch.float, device=device, requires_grad=False)

    for i, (x, y) in enumerate(tqdm(dataset)):
        X[i] = x.to(device)
        Y[i] = y

    return X, Y


def _load_CIFAR_from_dir(dir: str) -> Dict[str, datasets.CIFAR10]:
    trainset = datasets.CIFAR10(
        root=dir, train=True, download=True, transform=_CustomTransforms()
    )
    testset = datasets.CIFAR10(
        root=dir, train=False, download=True, transform=_CustomTransforms()
    )

    trainset, valset = _train_val_split(trainset, train_ratio=0.9, deterministic=True)

    return {"train": trainset, "val": valset, "test": testset}


def load_CIFAR(
    dir: str, load_into_device: Optional[str] = None
) -> Dict[str, MemoryDataset]:
    datasets = _load_CIFAR_from_dir(dir)
    if load_into_device is not None or load_into_device != "cpu":
        for key in datasets:
            X, Y = _move_dataset_to_device(datasets[key], load_into_device)
            datasets[key] = MemoryDataset(X, Y, None)

    return datasets
