import torch
from torch import nn, ones_like, zeros_like
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(object):
    def __init__(self, fill_pix=None, num_classes=10):
        self.fill_pix = fill_pix
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred):
        """
        Args
        ----------
        y_true: torch.tensor (n, 1, 224, 224)
        y_pred: torch.tensor (n, c, 224, 224)
        """
        # place mask over placeholder pixels (loss do not propagate through them)
        mask = torch.where(
            y_true == self.fill_pix, zeros_like(y_true), ones_like(y_true)
        )

        y_true = (y_true * mask).squeeze(dim=1)
        # one-hot encode labels
        y_true_one_hot = (
            nn.functional.one_hot(y_true.long(), num_classes=self.num_classes)
            .float()
            .permute(0, 3, 1, 2)
        )

        y_pred = y_pred * mask
        focal_loss = sigmoid_focal_loss(y_pred, y_true_one_hot, reduction="none")
        loss = (focal_loss * mask).sum() / (mask.sum() + 1e-16)

        return loss


class CELoss(object):
    def __init__(self, fill_pix=None):
        ignore_index = fill_pix if fill_pix is not None else -100
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def __call__(self, y_true, y_pred):
        """
        Args
        ----------
        y_true: torch.tensor (n, 1, 224, 224) or (n, 1)
        y_pred: torch.tensor (n, c, 224, 224) or (n, c)
        """
        if len(y_true.shape) == 4:
            y_true = y_true.squeeze(dim=1)
        loss = self.CE_loss(y_pred, y_true.long())
        return loss
