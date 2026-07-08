import torch.nn as nn
import torch.nn.functional as F

from matbind.model.components.resnet1d.layers import (
    AvgPool1dSamePadding,
    Conv1dSamePadding,
)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_function,
        use_projection=False,
        activation=F.relu,
    ):
        super().__init__()

        self.use_projection = use_projection

        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            norm_layer_function(out_channels),
        )

        self.conv1 = nn.Sequential(
            Conv1dSamePadding(in_channels, out_channels, kernel_size, stride, bias=False),
            norm_layer_function(out_channels),
        )

        self.conv2 = nn.Sequential(
            Conv1dSamePadding(out_channels, out_channels, kernel_size, 1, bias=False),
            norm_layer_function(out_channels),
        )

        self.activation = activation

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.projection(shortcut)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        return self.activation(x + shortcut)


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_function,
        use_projection=False,
        activation=F.relu,
        resnetd_shortcut=False,
    ):
        super().__init__()

        self.use_projection = use_projection
        self.activation = activation

        if use_projection:
            self.projection = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer_function(out_channels * self.expansion),
            )
            if resnetd_shortcut:
                self.projection = nn.Sequential(
                    AvgPool1dSamePadding(
                        kernel_size=2,
                        stride=stride,
                    ),
                    nn.Conv1d(
                        in_channels,
                        out_channels * self.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    norm_layer_function(out_channels * self.expansion),
                )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            norm_layer_function(out_channels),
        )

        self.conv2 = nn.Sequential(
            Conv1dSamePadding(out_channels, out_channels, kernel_size, stride, bias=False),
            norm_layer_function(out_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            norm_layer_function(out_channels * self.expansion),
        )

    def forward(self, x):
        shortcut = x

        if self.use_projection:
            shortcut = self.projection(shortcut)

        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)

        out += shortcut
        return self.activation(out)


class ResNetBlockGroup(nn.Module):
    def __init__(
        self,
        block_type: str,
        number_of_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_layer_function,
        activation: nn.Module = F.relu,
    ):
        super().__init__()
        if block_type not in ["residual", "bottleneck"]:
            raise ValueError(f"block_type must be one of ['residual', 'bottleneck'], got {block_type}")

        if block_type == "residual":
            block = ResidualBlock
            expansion = 1

        if block_type == "bottleneck":
            block = BottleneckBlock
            expansion = BottleneckBlock.expansion

        self.layers = nn.ModuleList()

        first_block = block(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm_layer_function,
            True,  # always use projection for the first block
            activation,
        )

        self.layers.append(first_block)

        for _ in range(1, number_of_blocks):
            layer = block(
                out_channels * expansion,
                out_channels,
                kernel_size,
                1,  # stride
                norm_layer_function,
                False,  # use projection
                activation,
            )

            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
