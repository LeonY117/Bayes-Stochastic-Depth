from typing import Optional, Tuple
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms


class MemoryDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        unified_transform: Optional[transforms.Compose] = None,
        x_transform: Optional[transforms.Compose] = None,
        y_transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Loads dataset from memory

        Args
        ------
        X : Tensor with shape (n, 3, H, W)
        Y : Tensor with shape (n, 3, H, W) or (n)
        transforms : transforms.Compose([...])
        """
        self.X = X
        self.Y = Y
        self.unified_transform = unified_transform

        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x, y = self.X[idx], self.Y[idx]
        if self.unified_transform:
            # x, y = self.transform((x, y))
            return self.unified_transform(x, y)  # to be consistent with torch

        if self.x_transform:
            x = self.x_transform(x)
        if self.y_transform:
            y = self.y_transform(y)
        return x, y
