import torch

from torch import Tensor
from typing import Optional, Tuple


def bayes_forward(
    net,
    X: torch.tensor,
    T: int,
    mode: Optional[str] = "all",
    buffer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Performs T forward passes with stochastic regularisation

    Args
    -----------
    net: nn.Module
    X  : torch.tensor (c x W x H), a single input image
    T  : int, indicating number of repeated forwards passes
    mode: str, indicating which bayesian mode to use
    buffer (optional): torch.tensor (T x c x W x H) buffer

    Returns
    -----------
    y_softmax            : torch.tensor (c x W x H)
    y_pred               : torch.tensor (W x H)
    y_pred_std_per_class : torch.tensor (c x W x H)
    y_pred_std_avg       : torch.tensor (W x H)
    """
    assert T > 0

    if buffer is None:
        buffer = X.unsqueeze(0).repeat(T, 1, 1, 1)
    else:
        for i in range(T):
            # write image to buffer
            buffer[i] = X

    with torch.no_grad():
        net.eval()
        net.set_bayes_mode(True, mode)
        y_logits = net(buffer)  # (T x c x W x H)

    # Average the softmax (note that the resultant vectors are not normalised)
    # y_logits = y_pred_raw.mean(dim=0)  # (T x c x W x H)
    y_softmax = y_logits.softmax(dim=1)  # (T x c x W x H)
    y_softmax_avg = y_softmax.mean(dim=0)  # (c x W x H)
    # Take max prob as prediction
    y_pred = torch.argmax(y_softmax_avg, dim=0).to(torch.int)  # (W x H)
    # Per class uncertainty
    y_pred_std_per_class = y_softmax.var(dim=0)  # (c x W x H)
    # Average uncertainty over classes
    y_pred_std_avg = y_pred_std_per_class.mean(dim=0)  # (W x H)

    return y_softmax_avg, y_pred, y_pred_std_per_class, y_pred_std_avg


def bayes_eval(
    net,
    X: Tensor,
    T: int,
    mode: Optional[str] = "all",
    buffer: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs T forward passes with dropout layers, returns prediction

    Args
    -----------
    net: nn.Module
    X  : torch.tensor (c x W x H), a single input image
    T  : int, indicating number of repeated forwards passes
    mode: str, indicating which bayesian mode to use
    buffer (optional): torch.tensor(T x c x W x H) buffer

    Returns
    -----------
    y_logits             : torch.tensor (c x W x H)
    y_pred               : torch.tensor (W x H)
    """
    assert T >= 0

    with torch.no_grad():
        net.eval()
        if T == 0:
            net.set_bayes_mode(False, "all")
            buffer = X.unsqueeze(0)  # (1 x c x W x H)
        elif T > 0:
            net.set_bayes_mode(True, mode)
            # write image to buffer
            if buffer is None:
                buffer = X.unsqueeze(0).repeat(T, 1, 1, 1)
            else:
                for i in range(T):
                    buffer[i] = X

    y_pred_raw = net(buffer)  # (T x c x W x H)
    y_logits = y_pred_raw.mean(dim=0)  # (c x W x H)
    y_pred = torch.argmax(y_logits, dim=0).to(torch.int8)  # (W x H)

    return y_logits, y_pred
