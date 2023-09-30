import torch

from typing import Tuple
from prettytable import PrettyTable
from torch import nn
from fvcore.nn import FlopCountAnalysis

__all__ = ["count_parameters", "calculate_storage_in_mb", "count_FLOPS"]


# helper functions for counting parameters and storage
def count_parameters(model, print_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if print_table:
        print(table)
    return total_params


def calculate_storage_in_mb(model, print_buffer=True):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    if print_buffer:
        print(f"Buffer size: {buffer_size/1024**2:.3f} MB")

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def count_FLOPS(net: nn.Module, input_dim=Tuple[int, int]) -> float:
    h, w = input_dim
    # we have to do 2 because 1 would cause a bug for batch norm
    device = net.parameters().__next__().device
    x_batch = torch.randn(2, 3, h, w).to(device)
    with torch.no_grad():
        net.eval()
        flops = FlopCountAnalysis(net, x_batch).total() / 2

    # print(f'Flops: {flops.total()/1e9:.3f} G')
    return flops
