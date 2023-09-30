from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import StochasticDepth

from dataclasses import dataclass

from .Bayesian_net import Bayesian_net

__all__ = [
    "resnet",
]


def conv3x3(
    inp: int, oup: int, stride: int = 1, groups: int = 1, padding: int = 1
) -> nn.Conv2d:
    """3x3 convolution"""
    return nn.Conv2d(
        inp,
        oup,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


def conv1x1(inp: int, oup: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        groups: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_p: Optional[float] = 0.0,
        sd_p: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.dropout_p = 0.0 if dropout_p is None else dropout_p
        self.sd_p = 0.0 if sd_p is None else sd_p

        self.conv1 = conv3x3(inp, oup, stride, groups)
        self.bn1 = norm_layer(oup)
        self.conv2 = conv3x3(oup, oup, 1, groups)
        self.bn2 = norm_layer(oup)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # to match dimensions when downsampling

        self.dropout = nn.Dropout2d(self.dropout_p)
        self.sd = StochasticDepth(self.sd_p, "row")

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.sd_p > 0.0:
            out = self.sd(out)  # sd: randomly zeros rows of out
        # add residual
        out += x
        if self.dropout_p > 0.0:
            out = self.dropout(out)  # dropout: randomly zeros units of out
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        groups: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_p: Optional[float] = 0.0,
        sd_p: Optional[float] = 0.0,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.dropout_p = 0.0 if dropout_p is None else dropout_p
        self.sd_p = 0.0 if sd_p is None else sd_p

        self.conv1 = conv1x1(inp, oup, 1, groups)
        self.bn1 = norm_layer(oup)
        # stride placed here rather than on 1x1, following torch implementation
        self.conv2 = conv3x3(oup, oup, stride, groups)
        self.bn2 = norm_layer(oup)
        self.conv3 = conv1x1(oup, oup * self.expansion, 1, groups)
        self.bn3 = norm_layer(oup * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # to match dimensions when downsampling
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.sd_p > 0.0:
            # sd: randomly zeros rows of out
            out = self.sd(out)
        # add residual
        out += x
        if self.dropout_p > 0.0:
            # dropout: randomly zeros units of out
            out = self.dropout(out)
        out = self.relu(out)
        return out


@dataclass
class _DropoutConfig:
    probs: List[float]

    @staticmethod
    def initiate_probs(stages: List[int], p: float, mode: str):
        if mode == "linear":
            return [p * (i + 1) / sum(stages) for i in range(sum(stages))]
        elif mode == "constant":
            return [p for _ in range(sum(stages))]
        elif mode == "classifier":
            # one extra prob for the classifer
            return [0.0 for _ in range(sum(stages))] + [p]
        elif mode == "none":
            return [0.0 for _ in range(sum(stages))]
        else:
            raise ValueError(f"Unknown dropout mode: {mode}")


@dataclass
class _SdConfig:
    probs: List[float]

    @staticmethod
    def initiate_probs(stages: List[int], p: float, mode: str):
        if mode == "linear":
            return [p * (i + 1) / sum(stages) for i in range(sum(stages))]
        elif mode == "constant":
            return [p for _ in range(sum(stages))]
        elif mode == "none":
            return [0.0 for _ in range(sum(stages))]
        else:
            raise ValueError(f"Unknown sd mode: {mode}")


class DropoutConfig(_DropoutConfig):
    def __init__(self, stages: List[int], p: float, mode: str):
        super().__init__(self.initiate_probs(stages, p, mode))


class SdConfig(_SdConfig):
    def __init__(self, stages: List[int], p: float, mode: str):
        super().__init__(self.initiate_probs(stages, p, mode))


class _ResNet(Bayesian_net):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        stages: List[int],
        num_classes: int = 10,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_cfg: Optional[DropoutConfig] = None,
        sd_cfg: Optional[SdConfig] = None,
    ) -> None:
        super().__init__()
        if not norm_layer:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.block = block

        self.stages = stages
        stage_widths = [64, 128, 256, 512]

        self.dropout_config = (
            drop_cfg if drop_cfg else DropoutConfig(stages, 0.0, "constant")
        )
        self.sd_config = sd_cfg if sd_cfg else SdConfig(stages, 0.0, "constant")

        self.conv1 = nn.Conv2d(
            3, stage_widths[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(stage_widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i, n in enumerate(stages):
            stride = 1 if i == 0 else 2

            in_width = 64 if i == 0 else stage_widths[i - 1] * block.expansion
            setattr(
                self,
                f"layer{i+1}",
                self._make_stage(
                    block,
                    in_width,
                    stage_widths[i],
                    n,
                    stride=stride,
                    block_id=sum(stages[:i]),
                ),
            )

        self.classifier_dropout = None
        if len(self.dropout_config.probs) > sum(self.stages):
            self.classifier_dropout = nn.Dropout(self.dropout_config.probs[-1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inp: int,
        oup: int,
        n: int,
        stride: int = 1,
        block_id: int = 0,  # block id of the first block
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inp != oup * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inp, oup * block.expansion, stride),
                norm_layer(oup * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inp,
                oup,
                stride,
                groups=1,
                downsample=downsample,
                norm_layer=norm_layer,
                dropout_p=self._get_dropout_p(block_id),
                sd_p=self._get_sd_p(block_id),
            )
        )

        for i in range(1, n):
            layers.append(
                block(
                    oup * block.expansion,
                    oup,
                    stride=1,
                    groups=1,
                    norm_layer=norm_layer,
                    dropout_p=self._get_dropout_p(block_id + i),
                    sd_p=self._get_sd_p(block_id + i),
                )
            )

        return nn.Sequential(*layers)

    def _get_dropout_p(self, id: int) -> float:
        return self.dropout_config.probs[id]

    def _get_sd_p(self, id: int) -> float:
        return self.sd_config.probs[id]

    def expected_layers(self) -> int:
        """
        Compute the expected depth with stochastic depth
        """
        sd_config = self.sd_config
        num_layers = sum([1 - p for p in sd_config.probs])
        # multiply by 2 because conv + bn, +2 for first conv and bn
        num_layers = num_layers * 2 + 2
        return num_layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.stages)):
            x = getattr(self, f"layer{i+1}")(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.classifier_dropout is not None:
            self.classifier_dropout(x)
        x = self.fc(x)
        return x


def resnet(
    resnet_name: str,
    num_classes: int = 10,
    dropout_mode: str = "none",
    dropout_p: float = 0.0,
    sd_mode: str = "none",
    sd_p: float = 0.0,
) -> _ResNet:
    "Wrapper class to parse dropout and sd configs correctly"

    if dropout_mode != "none" and sd_mode != "none":
        print("Warning: both dropout and sd are enabled. ")

    if resnet_name not in resnet_configs:
        raise ValueError(f"Unknown resnet name: {resnet_name}")

    stages, block = resnet_configs[resnet_name]

    dropout_config = DropoutConfig(stages, dropout_p, dropout_mode)
    sd_config = SdConfig(stages, sd_p, sd_mode)

    return _ResNet(
        block, stages, num_classes, drop_cfg=dropout_config, sd_cfg=sd_config
    )


resnet_configs = {
    "resnet18": ([2, 2, 2, 2], BasicBlock),
    "resnet34": ([3, 4, 6, 3], BasicBlock),
    "resnet50": ([3, 4, 6, 3], Bottleneck),
    "resnet101": ([3, 4, 23, 3], Bottleneck),
    "resnet152": ([3, 8, 36, 3], Bottleneck),
}
