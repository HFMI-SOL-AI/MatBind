import torch
from torch import Tensor, nn

from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, context_length: int):
        super().__init__()
        self.vocab_size = 30522
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, width))
        nn.init.normal_(self.positional_embedding, std=0.02)
        self.ln_final = LayerNorm(width)
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, self._build_attention_mask(context_length)) for _ in range(layers)])
        self.context_length = context_length

    def _build_attention_mask(self, context_length: int) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode(self, x: torch.Tensor):
        return self.resblocks(x)

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, padding_mask = x
        dtype = self.token_embedding.weight.dtype
        text_embedding = self.token_embedding(token_ids).type(dtype)  # [batch_size, n_ctx, d_model]

        text_embedding = text_embedding + self.positional_embedding.type(dtype)
        text_embedding = text_embedding.permute(1, 0, 2)  # NLD -> LND
        text_embedding = self.encode(text_embedding)
        text_embedding = text_embedding.permute(1, 0, 2)  # LND -> NLD
        text_embedding = self.ln_final(text_embedding).type(dtype)

        # x.shape = [batch_size, n_ctx, width]
        # take features from the sep embedding (end token), infer position from padding mask
        sep_pos = padding_mask.sum(dim=1) - 1  # [batch_size]
        batch_size = text_embedding.shape[0]
        return text_embedding[torch.arange(batch_size, device=text_embedding.device), sep_pos]
