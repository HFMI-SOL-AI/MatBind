import numpy as np
import torch
import torch.nn as nn


class LearnableLogitScaling(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = 1 / logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class Postprocessor(nn.Module):
    def __init__(
        self, logscaling, logit_scale_init: float = 0.07, learnable: bool = True, max_logit_scale: float = 100
    ) -> None:
        super().__init__()
        if logscaling:
            self.log_scalor = LearnableLogitScaling(
                logit_scale_init=logit_scale_init,
                learnable=learnable,
                max_logit_scale=max_logit_scale,
            )
            self.processor = nn.Sequential(
                Normalize(dim=-1),
                self.log_scalor,
            )
        else:
            self.log_scalor = None
            self.processor = Normalize(dim=-1)

    def forward(self, x):
        return self.processor(x)
