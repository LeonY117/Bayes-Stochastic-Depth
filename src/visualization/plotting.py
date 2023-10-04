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


def show_cifar_images(
    dataset: Dataset,
    grid_size: Tuple[int, int],
    show_labels: Optional[bool] = True,
    preds: Optional[List[int]] = None,
    indices: Optional[List[int]] = None,
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
            count = i * grid_size[0] + j
            if indices is not None:
                idx = indices[count]
            else:
                idx = count
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
