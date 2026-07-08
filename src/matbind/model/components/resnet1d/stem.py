import torch.nn as nn
import torch.nn.functional as F

from matbind.model.components.resnet1d.layers import (
    Conv1dSamePadding,
    MaxPool1DWithSamePadding,
)


class ResNet1dStem(nn.Module):
    def __init__(
        self,
        input_channels: int,
        norm_layer_function,
        stem_type: str = "v0",
        scale_stem: bool = True,
        depth_multiplier: float = 1.0,
        activation: nn.Module = F.relu,
        use_max_pool: bool = True,
    ):
        super().__init__()
        if (stem_type := stem_type.lower()) not in ["v0", "v1"]:
            raise ValueError(f"stem_type must be one of ['v0', 'v1'], got {stem_type}")

        self.stem_type = stem_type
        self.activation = activation
        self.use_max_pool = use_max_pool
        stem_depth_multiplier = depth_multiplier if scale_stem else 1.0

        self.out_channels = int(64 * stem_depth_multiplier)

        if stem_type == "v0":
            self.conv1 = nn.Sequential(
                Conv1dSamePadding(
                    input_channels,
                    self.out_channels,
                    kernel_size=7,
                    stride=2,
                    bias=False,
                ),
                norm_layer_function(self.out_channels),
            )

        if stem_type == "v1":
            filters = int(32 * stem_depth_multiplier)
            self.conv1 = nn.Sequential(
                Conv1dSamePadding(
                    input_channels,
                    filters,
                    kernel_size=3,
                    stride=2,
                    bias=False,
                ),
                norm_layer_function(filters),
            )
            self.conv2 = nn.Sequential(
                Conv1dSamePadding(
                    filters,
                    filters,
                    kernel_size=3,
                    stride=1,
                    bias=False,
                ),
                norm_layer_function(filters),
            )
            self.conv3 = nn.Sequential(
                Conv1dSamePadding(
                    filters,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    bias=False,
                ),
                norm_layer_function(self.out_channels),
            )

        if use_max_pool:
            self.last_layer = MaxPool1DWithSamePadding(kernel_size=3, stride=2)
        else:
            self.last_layer = nn.Sequential(
                Conv1dSamePadding(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    bias=False,
                ),
                norm_layer_function(self.out_channels),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        if self.stem_type == "v1":
            x = self.conv2(x)
            x = self.activation(x)

            x = self.conv3(x)
            x = self.activation(x)

        x = self.last_layer(x)

        if self.use_max_pool:
            return x

        return self.activation(x)
