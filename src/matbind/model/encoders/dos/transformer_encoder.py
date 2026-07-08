from typing import Literal

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from matbind.utils import rename_keys_with_prefix


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length, embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x has shape (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_layer: int,
        num_header: int,
        mixed_precision: bool,
        activation: Literal["relu", "gelu"] = "relu",
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.energy_embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, bias=False)
        self.density_embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, bias=False)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim=embedding_dim, max_seq_length=2001)

        self.down_layer = nn.Linear(2, 1, bias=False)

        self.dtype = torch.bfloat16 if mixed_precision else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_header,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation=activation,
            dtype=self.dtype,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        # load checkpoint if provided
        if ckpt_path is not None:
            self.load_state_dict(
                rename_keys_with_prefix(torch.load(ckpt_path)["state_dict"], "net."),
                strict=True,
            )
            if freeze_encoder:
                for param in self.parameters():
                    param.requires_grad = False
        else:
            self._init()
        # raise warning if freeze_encoder is True but ckpt_path is None
        if freeze_encoder and ckpt_path is None:
            raise Warning("freeze_encoder is True but ckpt_path is None")

    def _init(self):
        nn.init.xavier_uniform_(self.energy_embedding.weight)
        nn.init.xavier_uniform_(self.density_embedding.weight)

    def encode(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        density, energy, mask = inputs
        assert not torch.any(torch.isnan(density))
        assert not torch.any(torch.isnan(energy))
        assert density.shape == energy.shape
        assert density.shape[0] == mask.shape[0]
        N, S, _ = density.shape
        embedded_energy = self.energy_embedding(energy)
        assert embedded_energy.shape == (N, S, self.embedding_dim)
        assert not torch.any(torch.isnan(embedded_energy))

        embedded_density = self.density_embedding(density)
        assert embedded_density.shape == (N, S, self.embedding_dim)
        assert not torch.any(torch.isnan(embedded_density))

        concat = torch.stack([embedded_energy, embedded_density], dim=-1)
        assert concat.shape == (N, S, self.embedding_dim, 2)
        assert not torch.any(torch.isnan(concat))

        x = self.down_layer(concat).squeeze(-1)
        assert x.shape == (N, S, self.embedding_dim)
        assert not torch.any(torch.isnan(x))

        x = self.positional_encoding(x)
        x = x * mask.float().unsqueeze(-1)

        # Apply gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            # Checkpoint each transformer layer
            x_input = x.to(self.dtype)
            for layer in self.transformer.layers:
                x_input = checkpoint.checkpoint(layer, x_input, src_key_padding_mask=~mask.bool(), use_reentrant=False)
            outputs = x_input
        else:
            outputs = self.transformer(x.to(self.dtype), src_key_padding_mask=~mask.bool())

        assert not torch.any(torch.isnan(outputs))

        output = (outputs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        assert not torch.any(torch.isnan(output))
        return output

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        return self.encode(inputs)
