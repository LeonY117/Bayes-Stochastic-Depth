import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, List, Optional
from torch.utils.data import Dataset

__all__ = ["imshow_image", "unnormalize", "show_cifar_images"]


def imshow_image(img, inv_norm: bool = True) -> None:
    if inv_norm:
        img = unnormalize(img)
    plt.imshow(img.cpu().permute(1, 2, 0))
    plt.axis("off")


def unnormalize(image: torch.tensor) -> torch.tensor:
    """
    Reverses imageNet Normalization to [0, 1], (for visualization purposes)
    """
    mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    reverse_normalize = transforms.Normalize(mean, std)

    return torch.clip(reverse_normalize(image), 0, 1)


def show_cifar_images(
    grid_size: Tuple[int, int],
    show_labels: bool = True,
    dataset: Dataset = None,
    preds: List[int] = None,
):
    """
    Show a grid of images from CIFAR10 dataset
    """
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    if preds is not None:
        assert len(preds) == grid_size[0] * grid_size[1]
    if dataset is None:
        raise ValueError("dataset cannot be None")
    n1, n2 = grid_size
    plt.subplots(n1, n2, figsize=(n2 * 1.5, n1 * 1.5))
    for i in range(n2):
        for j in range(n1):
            idx = i * grid_size[0] + j
            img, label = dataset[idx]
            label = int(label)
            plt.subplot(n1, n2, idx + 1)
            imshow_image(img)
            if show_labels:
                plt.title(classes[label])
            if preds is not None:
                if label != preds[idx]:
                    color = "red"
                plt.title(f"{classes[label]}-{classes[preds[idx]]}", color=color)
    plt.tight_layout()
    plt.show()
