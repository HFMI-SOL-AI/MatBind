from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path

import torch

from matbind.model.components.resnet1d.resnet1d import ResNet1d
from matbind.model.encoders.crystal_structure.csgcnn import CSGCNN
from matbind.model.encoders.dos.transformer_encoder import TransformerEncoder
from matbind.model.encoders.pxrd.resnet_encoder import ResNetEncoder
from matbind.model.encoders.text.bert_encoder import TextEncoder
from matbind.model.matbind_model import MatBind

ENCODER_EMBEDDING_SIZE = 256


class ProjectionHeadConfig:
    def __init__(self, is_on: bool, freeze: bool, init_kwargs: dict):
        self.is_on = is_on
        self.freeze = freeze
        self.init_kwargs = init_kwargs


class EncoderConfig:
    def __init__(
        self,
        encoder: Callable[[], torch.nn.Module],
        projection_head: ProjectionHeadConfig,
        postprocessor: torch.nn.Module | None,
    ):
        self.encoder = encoder
        self.projection_head = projection_head
        self.postprocessor = postprocessor


dos_encoder_config = EncoderConfig(
    encoder=lambda: TransformerEncoder(
        input_dim=1,
        embedding_dim=512,
        num_layer=2,
        num_header=8,
        mixed_precision=False,
    ),
    projection_head=ProjectionHeadConfig(
        is_on=True, freeze=False, init_kwargs={"dims": [512, ENCODER_EMBEDDING_SIZE], "activation": "LeakyReLU"}
    ),
    postprocessor=None,
)

crystal_structure_encoder_config = EncoderConfig(
    encoder=lambda: CSGCNN(
        num_node_features=96,
        num_edge_features=41,
        hidden_channels=128,
        num_layers=6,
    ),
    projection_head=ProjectionHeadConfig(
        is_on=True, freeze=False, init_kwargs={"dims": [128, ENCODER_EMBEDDING_SIZE], "activation": "LeakyReLU"}
    ),
    postprocessor=None,
)

text_encoder_config = EncoderConfig(
    encoder=lambda: TextEncoder(
        config="/p/project1/solai/datasets/matbert-base-cased/config.json",
        checkpoint_path="/p/project1/solai/datasets/matbert-base-cased",
        max_length=512,
    ),
    projection_head=ProjectionHeadConfig(
        is_on=True, freeze=False, init_kwargs={"dims": [768, ENCODER_EMBEDDING_SIZE], "activation": "LeakyReLU"}
    ),
    postprocessor=None,
)

pxrd_encoder_config = EncoderConfig(
    encoder=lambda: ResNetEncoder(
        resnet=ResNet1d(
            model_id=777,
            in_channels=1,
            kernel_size=9,
            stride=4,
            use_group_norm=True,
        ),
        freeze=False,
    ),
    projection_head=ProjectionHeadConfig(
        is_on=True, freeze=False, init_kwargs={"dims": [512, ENCODER_EMBEDDING_SIZE], "activation": "LeakyReLU"}
    ),
    postprocessor=None,
)

ENCODERS = {
    "dos": dos_encoder_config,
    "crystal_structure": crystal_structure_encoder_config,
    "text": text_encoder_config,
    "pxrd": pxrd_encoder_config,
}


def compare_state_dicts(ckpt1_statedict: dict, ckpt2_statedict: dict):
    for (old_k, old_v), (new_k, new_v) in zip_longest(ckpt1_statedict.items(), ckpt2_statedict.items(), fillvalue=(None, None)):
        old_str = f"{old_k}: {tuple(old_v.shape)}" if old_k else "MISSING"
        new_str = f"{new_k}: {tuple(new_v.shape)}" if new_k else "MISSING"
        print(f"{old_str} -> {new_str}")


def compare_encoders(ckpt1_statedict: dict, ckpt2_statedict: dict, modality: str):
    old_encoder = {k: v for k, v in ckpt1_statedict.items() if modality in k}
    new_encoder = {k: v for k, v in ckpt2_statedict.items() if modality in k}

    print(f"{modality.capitalize()} Encoder Comparison:")
    compare_state_dicts(old_encoder, new_encoder)


def compare_all_encoders(ckpt1_statedict: dict, ckpt2_statedict: dict, modalities: list[str]):
    for modality in modalities:
        compare_encoders(ckpt1_statedict, ckpt2_statedict, modality)


def main():
    current_model = MatBind(modalities=ENCODERS.keys(), encoders=ENCODERS).state_dict()  # type: ignore
    old_ckpt_path = Path("/p/project1/solai/datasets/checkpoints/pxrd_text_crys_dos_20250207_2223/epoch=088_valid_loss=0.67.ckpt")
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")["state_dict"]

    current_model = dict(sorted(current_model.items()))
    old_ckpt = dict(sorted(old_ckpt.items()))
    # compare_all_encoders(old_ckpt, current_model, modalities=list(ENCODERS.keys()))
    compare_state_dicts(old_ckpt, current_model)


if __name__ == "__main__":
    main()
