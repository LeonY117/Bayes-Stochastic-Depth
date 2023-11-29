import torch
from torch import Tensor
from typing import Dict, List


class AccuracyMetric(object):
    def __init__(self, metrics=["acc"], classes: List[str] = None) -> None:
        """
        Args
        ----------
        metrics: list of metrics to compute, options are ["acc", "iou"]
        classes: list containing class names
        """
        self.compute_acc = "acc" in metrics
        self.compute_iou = "iou" in metrics
        self.classes = classes

    def __call__(self, confusion_matrix: Tensor) -> Dict[str, float]:
        """
        Args
        ---------
        confusion_matrix: Tensor (c x c)
        n: int

        Returns
        ---------
        metrics: Dict[str, float] containing metrics
        """
        cm = confusion_matrix
        num_classes = cm.shape[0]
        metrics = {}

        TPs = torch.diagonal(cm, 0)
        FPs = torch.sum(cm, dim=0) - TPs
        FNs = torch.sum(cm, dim=1) - TPs

        if self.compute_acc:
            accs = TPs / (TPs + FNs)
            avg_acc = accs.mean()
            for c in range(num_classes):
                metrics[f"acc/{self.classes[c]}"] = accs[c].item()
            metrics["acc/avg"] = avg_acc.item()
            metrics["acc/global"] = TPs.sum().item() / (cm.sum().item() + 1e-16)

        if self.compute_iou:
            ious = TPs / (FNs + FPs + TPs)
            miou = ious.mean()

            for c in range(num_classes):
                metrics[f"iou/{self.classes[c]}"] = ious[c].item()
            metrics["iou/mean"] = miou.item()

        return metrics


def get_confusion_matrix(y_gt: Tensor, y_pred: Tensor, num_classes: int) -> Tensor:
    """
    Compute the bins to update the confusion matrix with, code adapted from torchmetrics
    Args
    ----------
    y_gt: torch.tensor (n, 1, 224, 224) or (n)
    y_pred: torch.tensor (n, 1, 224, 224) or (n)
    num_classes: int

    Returns
    ----------
    confusion_matrix: torch.tensor(num_classes, num_classes)
    """

    unique_mapping = y_gt.to(torch.long) * num_classes + y_pred.to(torch.long)
    bins = torch.bincount(
        unique_mapping.flatten(), minlength=num_classes * (num_classes + 1)
    )

    return bins[: num_classes**2].reshape(num_classes, num_classes)
