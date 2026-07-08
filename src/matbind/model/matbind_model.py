from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from matbind.model.components.projection_head import ProjectionHead
from matbind.model.components.postprocessing import Postprocessor
from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


class ProjectionHeadInitKwargs(Protocol):
    dims: list[int]
    activation: str


class ProjectionHeadConfig(Protocol):
    is_on: bool
    freeze: bool
    init_kwargs: ProjectionHeadInitKwargs


class EncoderConfig(Protocol):
    encoder: Callable[[], nn.Module]
    projection_head: ProjectionHeadConfig
    postprocessor: nn.Module


EncoderConfigDict = dict[str, EncoderConfig]

Forwardable = Tensor | Data
EmbeddingDict = dict[str, Tensor]


class MatBind(nn.Module):
    def __init__(
        self,
        modalities: list[str],
        encoders: EncoderConfigDict,
    ) -> None:
        super().__init__()

        LOGGER.info(f"Building Model with following modalities: {modalities}")

        if not set(encoders.keys()).issuperset(set(modalities)):
            raise ValueError(
                f"All modalities must have an encoder configuration, but {set(encoders.keys()).difference(set(modalities))} are not avaliable."
            )

        dict_encoders = {}
        dict_projection_heads = {}
        postprocessors = {}

        for modality in modalities:
            encoder, projection_head, postprocessor = self.get_encoder_and_projection_head(modality, encoders)
            if encoder:
                dict_encoders[modality] = encoder
            if projection_head:
                dict_projection_heads[modality] = projection_head
            if postprocessor:
                postprocessors[modality] = postprocessor

        self.dict_encoders = nn.ModuleDict(dict_encoders)
        self.dict_projection_heads = nn.ModuleDict(dict_projection_heads)
        self.postprocessors = nn.ModuleDict(postprocessors)
        self.modalities = modalities

    def get_encoder_and_projection_head(
        self, modality: str, encoders: EncoderConfigDict
    ) -> tuple[nn.Module | None, nn.Module | None, nn.Module]:
        encoder_config = encoders[modality]

        encoder = encoder_config.encoder()
        projection_head_config = encoder_config.projection_head
        if encoder_config.postprocessor is not None:
            postprocessor = Postprocessor(**encoder_config.postprocessor)  # type: ignore
        else:
            postprocessor = nn.Identity()

        if not projection_head_config.is_on:
            return encoder, None, postprocessor

        projection_head = ProjectionHead(**projection_head_config.init_kwargs)  # type: ignore

        if projection_head_config.freeze:
            for param in projection_head.parameters():
                param.requires_grad = False

        return encoder, projection_head, postprocessor

    def encode(
        self,
        modality: str,
        input_data: Forwardable,
        with_projection: bool = False,
    ) -> torch.Tensor:
        encoder = self.dict_encoders[modality]
        postprocessor = self.postprocessors[modality]
        embedding = encoder(input_data)

        if not with_projection:
            return embedding

        if modality not in self.dict_projection_heads:
            return postprocessor(embedding)

        projection_head = self.dict_projection_heads[modality]
        embedding = projection_head(embedding)
        return postprocessor(embedding)

    def forward(
        self,
        input_data: dict[str, Forwardable],
    ) -> EmbeddingDict:
        embeddings = {}

        for modality, data in input_data.items():
            LOGGER.debug(f"MatBind Forwarding Modality {modality}")
            embeddings[modality] = self.encode(modality, data, with_projection=True)

        return embeddings
