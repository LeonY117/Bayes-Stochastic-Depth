import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from typing import Dict, Optional, Tuple
from torch import Tensor

from tqdm import tqdm

from .memory_dataset import MemoryDataset

__all__ = ["get_dataset"]


class _PreprocessTransforms(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x):
        # x, y = sample
        x = self.to_tensor(x)
        x = x.to(torch.float)
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
    Takes a dataset object and loads everything onto device
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
    """
    Loads CIFAR10 from the given directory, and splits it into train, val, and test sets. No transforms are applied.
    """
    trainset = datasets.CIFAR10(
        root=dir, train=True, download=True, transform=_PreprocessTransforms()
    )
    valset = datasets.CIFAR10(
        root=dir, train=False, download=True, transform=_PreprocessTransforms()
    )

    # trainset, valset = _train_val_split(trainset, train_ratio=0.9, deterministic=True)

    return {"train": trainset, "val": valset}


# def load_CIFAR(
#     dir: str, load_into_device: Optional[str] = None
# ) -> Dict[str, MemoryDataset]:
#     datasets = _load_CIFAR_from_dir(dir)
#     if load_into_device is not None or load_into_device != "cpu":
#         for key in datasets:
#             X, Y = _move_dataset_to_device(datasets[key], load_into_device)
#             datasets[key] = MemoryDataset(X, Y, None)

#     return datasets


def get_dataset(
    dataset_name: str, dir: str, device: Optional[str] = None
) -> Dict[str, MemoryDataset]:
    assert dataset_name in ["cifar10"]

    if dataset_name == "cifar10":
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        test_transforms = transforms.Compose(
            [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        )

        datasets = _load_CIFAR_from_dir(dir)
        print('loading dataset into memory...')
        X, Y = _move_dataset_to_device(datasets["train"], device)
        datasets["train"] = MemoryDataset(X, Y, x_transform=train_transforms)

        X, Y = _move_dataset_to_device(datasets["val"], device)
        datasets["val"] = MemoryDataset(X, Y, x_transform=test_transforms)

        # X, Y = _move_dataset_to_device(datasets["test"], device)
        # datasets["test"] = MemoryDataset(X, Y, x_transform=test_transforms)

    return datasets
