import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForMaskedLM, BertConfig


def xavier_init(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    return model


class TextEncoder(nn.Module):
    def __init__(
        self,
        config: str,  # Path to config.json
        checkpoint_path: str,
        freeze_encoder: bool = False,
        # tokenizer=None,
        max_length: int = 512,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        # Load config from json file. TODO: Maybe this setting should also be handled by hydra
        bert_config = BertConfig.from_pretrained(config)
        bert_config.max_position_embeddings = max_length

        # Check if checkpoint_path is a Lightning checkpoint or standard pretrained model
        if checkpoint_path.endswith('.ckpt'):
            print(f"Loading from {checkpoint_path}")
            # Load from Lightning checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Extract BERT weights from Lightning checkpoint (typically under 'model.encoder' or similar)
            bert_state_dict = {}
            for key, value in state_dict.items():
                if 'model.dict_encoders.text.encoder.' in key:
                    # Remove 'model.dict_encoders.text.' prefix to match BERT model structure
                    new_key = key.replace('model.dict_encoders.text.encoder.', '')
                    bert_state_dict[new_key] = value
                elif key.startswith('bert.') or key.startswith('transformer.'):
                    bert_state_dict[key] = value

            # Initialize model from config, then load Lightning checkpoint weights
            self.encoder = AutoModelForMaskedLM.from_config(bert_config, trust_remote_code=True)
            if bert_state_dict:
                self.encoder.load_state_dict(bert_state_dict, strict=True)
        else:
            # Load from standard pretrained model path
            self.encoder = AutoModelForMaskedLM.from_pretrained(checkpoint_path, config=bert_config, trust_remote_code=True)

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # self.tokenizer = tokenizer
        self.max_length = max_length

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, attention_mask = x
        # Ensure input tensors have correct shape
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Truncate if necessary
        if token_ids.size(1) > self.max_length:
            token_ids = token_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        # Add token type ids
        token_type_ids = torch.zeros_like(token_ids)

        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        last_hidden_state = output.hidden_states[-1]

        return self._non_pad_token_embed_averaging(last_hidden_state, attention_mask)

    def _non_pad_token_embed_averaging(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_ = (last_hidden_state * attention_mask).sum(dim=1)
        norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
        return sum_ / norm
