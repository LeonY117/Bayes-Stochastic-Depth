import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset

# __all__ = ["imshow_image", "unnormalize", "show_cifar_images",]


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


def imshow_cifar(
    images,
    classes: List[str],
    grid_size: Tuple[int, int],
    labels: Optional[List[int]],
    preds: Optional[List[int]],
):
    """
    Show a grid of images from CIFAR10 dataset
    """

    n1, n2 = grid_size
    plt.subplots(n1, n2, figsize=(n2 * 1.5, n1 * 1.5))
    for i in range(n2):
        for j in range(n1):
            idx = i * grid_size[0] + j

            plt.subplot(n1, n2, idx + 1)
            imshow_image(images[idx])
            title = ""
            color = "black"
            if labels is not None:
                title += classes[labels[idx]]
            if preds is not None:
                title += classes[preds[idx]]
            if preds is not None and labels is not None:
                color = "red" if labels[idx] == preds[idx] else "green"
            plt.title(title, color=color)
    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history: Dict[str, List[float]]) -> None:
    """
    Plot loss history
    Args:
        loss_history: Dict[str, List[float]] = {"train": [], "val": []}
    """
    plt.plot(loss_history["train"], label="train")
    plt.plot(loss_history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def plot_acc_history(acc_history: Dict[str, List[float]]) -> None:
    """
    Plot accuracy history
    Args:
        acc_history: Dict[str, List[float]] = {"train": [], "val": []}
    """
    plt.plot(acc_history["train"], label="train")
    plt.plot(acc_history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()


def plot_per_class_acc_history(
    acc_history: Dict[str, List[float]], classes: List[str]
) -> None:
    """
    Plots per-class accuracy history
    Args:
        acc_history: Dict[str, List[float]] = {className: []}
        classes: List[str] containing class names
    """
    for i, cls in enumerate(classes):
        plt.plot(acc_history[cls], label=cls)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()


def plot_calibration():
    pass
