import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import CGConv, global_mean_pool


class CSGCNN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        num_layers: int,
        ckpt_path: str | None = None,
        freeze_encoder: bool = False,
        remove_head: bool = True,
        normal_layer: str = "layernorm",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.remove_head = remove_head
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Initial node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        # CGConv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        def normal_layer_fc(hidden_channels):
            if normal_layer == "layernorm":
                return nn.LayerNorm(hidden_channels)
            elif normal_layer == "groupnorm":
                return nn.GroupNorm(32, hidden_channels)
            elif normal_layer == "batchnorm":
                return nn.BatchNorm1d(hidden_channels, track_running_stats=False)
            else:
                raise ValueError(f"Unsupported normalization layer: {normal_layer}")

        for _ in range(self.num_layers):
            self.convs.append(
                CGConv(
                    hidden_channels,
                    num_edge_features,
                    bias=True,
                )
            )
            self.batch_norms.append(
                normal_layer_fc(
                    hidden_channels,
                )
            )
        if ckpt_path:
            self.load_pretrained(ckpt_path)
        if freeze_encoder:
            self.freeze_encoder()
        # self.output_layer = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels, 1),
        # )

    def _conv_layer_forward(self, i, x, edge_index, edge_attr):
        """Helper function for gradient checkpointing"""
        x = self.convs[i](x, edge_index, edge_attr)
        x = F.relu(x)
        if x.size(0) > 1:
            x = self.batch_norms[i](x)
        return x

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x.float(),
            data.edge_index,
            data.edge_attr.float(),
            data.batch,
        )
        # Initial node embedding
        x = self.node_embedding(x)

        # Graph convolutions with optional gradient checkpointing
        for i in range(len(self.convs)):
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(
                    self._conv_layer_forward,
                    i,
                    x,
                    edge_index,
                    edge_attr,
                    use_reentrant=False,
                )
            else:
                x = self._conv_layer_forward(i, x, edge_index, edge_attr)

        # Global pooling
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding

    def load_pretrained(self, ckpt_path):
        """
        Load pretrained weights from a file.
        """
        if not ckpt_path:
            return

        try:
            # Load the state dict
            state_dict = torch.load(ckpt_path, weights_only=True)

            # remove the output layer if needed
            if self.remove_head:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("output_layer")}

            # If it's a checkpoint file, extract just the model state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            encoder_state_dict={}
            for key, value in state_dict.items():
                if 'model.dict_encoders.crystal_structure.' in key:
                    # Remove 'model.dict_encoders.text.' prefix to match BERT model structure
                    new_key = key.replace('model.dict_encoders.crystal_structure.', '')
                    encoder_state_dict[new_key] = value

            # Load the state dict, allowing for missing or unexpected keys
            self.load_state_dict(encoder_state_dict, strict=True)
            print(f"Loaded pretrained model from {ckpt_path}")
        except Exception as e:
            print(f"Error loading pretrained model: {e!s}")

    def freeze_encoder(self):
        """
        Freeze the encoder part of the model (useful for fine-tuning).
        """
        for param in self.node_embedding.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        for bn in self.batch_norms:
            for param in bn.parameters():
                param.requires_grad = False
