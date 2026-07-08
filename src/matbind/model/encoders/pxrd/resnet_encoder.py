import torch
import torch.nn as nn

from matbind.model.components.resnet1d.resnet1d import ResNet1d
from matbind.utils.utils import rename_keys_with_prefix


class pXRDEncoder(nn.Module):
    def __init__(
        self,
        model_id: int | str = 10,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.resnet = ResNet1d(
            model_id=model_id,
            in_channels=1,
            use_group_norm=True,
            kernel_size=9,
            stride=4,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        if freeze_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if ckpt_path is not None:
            self.load_state_dict(rename_keys_with_prefix(torch.load(ckpt_path)["state_dict"]), strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return torch.mean(x, dim=-1)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        resnet: ResNet1d,
        ckpt_path: str | None = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.resnet = resnet
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if ckpt_path is not None:
            self.load_state_dict(
                rename_keys_with_prefix(torch.load(ckpt_path)["state_dict"]),
                strict=False,
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.pool(x)
        x = self.squeeze(x)
        return x
