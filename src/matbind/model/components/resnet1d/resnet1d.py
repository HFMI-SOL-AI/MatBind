import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from matbind.model.components.resnet1d.blocks import BottleneckBlock, ResNetBlockGroup
from matbind.model.components.resnet1d.specs import RESNET_SPECS
from matbind.model.components.resnet1d.stem import ResNet1dStem


class ResNet1d(nn.Module):
    def __init__(
        self,
        model_id: int | str,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        stem_type: str = "v0",
        scale_stem: bool = True,
        use_max_pool_in_stem: bool = True,
        depth_multiplier: float = 1.0,
        activation: nn.Module = F.relu,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        use_group_norm: bool = False,
        disable_normalization: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

        if model_id not in RESNET_SPECS:
            raise ValueError(f"Unsupported ResNet depth: {model_id}")

        if use_group_norm:

            def norm_layer_function(num_features):
                return nn.GroupNorm(32, num_features)
        else:

            def norm_layer_function(num_features):
                return nn.BatchNorm1d(num_features, momentum=norm_momentum, eps=norm_epsilon)

        if disable_normalization:

            def norm_layer_function(num_features):
                return nn.Identity(num_features)

        self.in_channels = in_channels

        self.resnet_stem = ResNet1dStem(
            in_channels,
            norm_layer_function,
            stem_type,
            scale_stem,
            depth_multiplier,
            activation,
            use_max_pool_in_stem,
        )
        spec = RESNET_SPECS[model_id]

        self.blockgroups = nn.ModuleList()
        in_channels = self.resnet_stem.out_channels

        for i, (block_type, out_channels, repetitions) in enumerate(spec):
            stride_ = 1 if i == 0 else stride

            out_channels = int(out_channels * depth_multiplier)
            blockgroup = ResNetBlockGroup(
                block_type,
                repetitions,
                in_channels,
                out_channels,
                kernel_size,
                stride_,
                norm_layer_function,
                activation,
            )

            self.blockgroups.append(blockgroup)

            in_channels = out_channels * BottleneckBlock.expansion if block_type == "bottleneck" else out_channels

        self.out_channels = in_channels

    def get_output_shape(
        self,
        input_length: int,
        following_module: nn.Module = None,
    ) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, input_length)
            dummy_output: torch.Tensor = self.forward(dummy_input)
            if following_module:
                dummy_output = following_module(dummy_output)

            return dummy_output.shape

    def forward(self, x) -> torch.Tensor:
        x = self.resnet_stem(x)
        for block in self.blockgroups:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
