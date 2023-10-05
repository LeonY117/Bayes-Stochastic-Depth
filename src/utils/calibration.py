from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from tqdm import tqdm
from .bayesian_utils import bayes_eval

__all__ = [
    "compute_calibration_errors",
    "_get_calibration",
    "_compute_calibration_statistics",
]


def _compute_calibration_statistics(total_counts, total_corrects, confs):
    """
    Computes the ECE and MCE statistics given the buckets
    """
    num_buckets = len(total_counts)
    n = total_counts.sum()

    ECE = 0
    MCE = 0

    for i in range(num_buckets):
        conf = confs[i]
        acc = total_corrects[i] / total_counts[i]
        calibration_error = total_counts[i] * np.absolute(conf - acc)

        ECE += calibration_error / n
        MCE = max(calibration_error / n, MCE)

    return ECE, MCE


def _get_calibration(
    net: nn.Module,
    dataloader: DataLoader,
    k: int,
    mode: Optional[str] = "all",
    num_buckets: Optional[int] = 10,
    fill_pix: Optional[int] = None,
):
    """
    Returns array of buckets for the reliability plot

    Args
    -----------
    net: nn.Module
    dataloader: torch.Dataloader
    k: set to 0 for regular forward pass, or larger than 0 for bayes forward pass
    mode: sets the mode of which bayesian inference to perform
    num_buckets: number of buckets to separate between 0 to 100 probabiliy

    Returns
    -----------
    total_counts: (num_buckets, ) np array, number of instances with confidence in the bucket
    total_corrects: (num_buckets, ) np array, number of correct instances with confidence in the bucket
    total_confs: (num_buckets, ) np array, the confidence score for each bucket by the network
    """
    total_counts = np.array([1e-16] * num_buckets, dtype=np.float32)
    total_corrects = np.array([0] * num_buckets, dtype=np.float32)
    total_confs = np.array([0] * num_buckets, dtype=np.float32)

    # loop through images and make predictions
    net.eval()
    for X_batch, y_batch in tqdm(dataloader):
        # force loop to be over single images
        for x, y in zip(X_batch, y_batch):
            # bayes forward pass
            if k == 0:
                net.set_bayes_mode(False, "all")
            elif k > 0:
                net.set_bayes_mode(True, mode)

            y_logits, y_pred = bayes_eval(net, x, k)
            y_softmax = y_logits.softmax(dim=0)

            counts, corrects, conf = _get_calibration_per_prediction(
                y, y_softmax, num_buckets, fill_pix
            )
            total_counts += counts
            total_corrects += corrects
            total_confs += conf

    # buckets = total_corrects / total_counts

    return total_counts, total_corrects, total_confs / total_counts


def _get_calibration_per_prediction(
    y_gt: Tensor,
    y_softmax: Tensor,
    num_buckets: int = 10,
    fill_pix: Optional[int] = None,
) -> Tuple[np.array, np.array, np.array]:
    """
    Computes reliability values for each image

    Args
    -----------
    y_gt: torch.tensor (1 x W x H)
    y_softmax: torch.tensor (c x W x H)
    num_buckets: int
    fill_pix: int, if not None, will ignore pixels with this value

    Returns
    -----------
    counts: np.array (num_buckets, ) counts in each bin
    correct: np.array (num_buckets, ) number of corrects in each bin
    conf: np.array (num_buckets, ) the average confidence p
    """
    if fill_pix is not None:
        mask = (y_gt != fill_pix).to(float)
    else:
        mask = torch.ones_like(y_gt).to(float)
    out_count = np.array([0 for _ in range(num_buckets)], dtype=np.float32)
    out_correct = np.array([0 for _ in range(num_buckets)], dtype=np.float32)
    out_conf = np.array([0 for _ in range(num_buckets)], dtype=np.float32)
    step = 1 / num_buckets
    # get probability and prediction
    y_prob, y_pred = torch.max(y_softmax, dim=0)
    indices = y_prob // step

    for i in range(num_buckets):
        idx_mask = mask * (indices == i)
        out_count[i] = (idx_mask).sum()
        out_correct[i] = (idx_mask * (y_pred == y_gt)).sum()
        out_conf[i] = (idx_mask * y_prob).sum().item()
    return out_count, out_correct, out_conf


def compute_calibration_errors(
    net: nn.Module,
    dataloader: DataLoader,
    k: int,
    mode: Optional[str] = "all",
    num_buckets: Optional[int] = 10,
) -> Tuple[float, float]:
    """
    Computes the ECE and MCE statistics given the buckets
    Args
    -----------
    net: nn.Module
    dataloader: torch.Dataloader
    k: set to 0 for regular forward pass, or larger than 0 for bayes forward pass
    mode: sets the mode of which bayesian inference to perform
    num_buckets: number of buckets to separate between 0 to 100 probabiliy

    Returns
    -----------
    ECE: (float) Expected Calibration Error
    MCE: (float) Maximum Calibration Error

    """
    total_counts, total_corrects, confs = _get_calibration(
        net, dataloader, k, mode, num_buckets
    )
    ECE, MCE = _compute_calibration_statistics(total_counts, total_corrects, confs)

    return ECE, MCE
