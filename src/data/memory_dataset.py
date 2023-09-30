from typing import Optional, Tuple
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms


class MemoryDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        transform: Optional[transforms.Compose] = None,
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
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            # x, y = self.transform((x, y))
            x, y = self.transform(x, y)  # to be consistent with torch
        return x, y
