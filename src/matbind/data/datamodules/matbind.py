import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.combined_loader import _LITERAL_SUPPORTED_MODES
from sklearn.model_selection import train_test_split

from matbind.data.datasets.builder import MultiModalDatasetBuilder
from matbind.data.loading import DatasetLoader, load_dataset
from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)

SupportedModes = Literal["max_size", "max_size_cycle"]


class MatBindDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        central_modality: str,
        modalities: list[str],
        frac_data: float,
        dataset_loader: DatasetLoader | None,
        dataset_builder: MultiModalDatasetBuilder,
        id_column: str,
        frac_train: float,
        batch_size: int,
        drop_last: bool,
        seed: int,
        combined_loader_mode: SupportedModes,
        **dataloader_kwargs: dict[str, Any],
    ):
        super().__init__()

        self.data_dir = Path(data_dir) if data_dir else None
        self.central_modality = central_modality
        self.modalities = modalities
        self.frac_data = frac_data
        self.dataset_loader = dataset_loader
        self.dataset_builder = dataset_builder
        self.id_column = id_column

        self.frac_train = frac_train
        self.drop_last = drop_last
        self.seed = seed
        self.batch_size = batch_size
        self.combined_loader_mode = combined_loader_mode
        self.dloader_kwargs = dataloader_kwargs

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        if self.data_dir is None:
            self.train_data = get_mock_data(
                self.central_modality,
                self.modalities,
                self.id_column,
            )
            self.val_data = get_mock_data(
                self.central_modality,
                self.modalities,
                self.id_column,
            )
            LOGGER.warning(f"No data_dir provided. Using mock data. Data Head:\n{self.train_data.head()}")
            return

        self.dataset = load_dataset(
            self.data_dir,
            self.dataset_loader,
        )
        LOGGER.info(f"Loaded dataframe has the following columns {self.dataset.columns}")
        self.dataset = self.dataset.dropna(subset=[self.central_modality])
        LOGGER.info(f"Loaded complete dataset with {len(self.dataset)} samples before splitting.")

        if not set(self.modalities).issubset(set(self.dataset.columns)):
            LOGGER.warning(
                f"The provided modalities={self.modalities} are not present in the dataset columns={self.dataset.columns}"
            )

        if self.frac_train == 0.0:
            n_samples = int(self.frac_data * len(self.dataset))
            LOGGER.info(
                f"Using n_samples=len(dataset)*frac_data={len(self.dataset)}*{self.frac_data}={n_samples} for prediction."
            )
            self.train_data = None
            self.val_data = self.dataset.head(n_samples)  # use head to maintain order
            self.check_data(self.val_data, "Prediction")
            return

        self.dataset = self.dataset.sample(frac=self.frac_data, random_state=self.seed)
        self.train_data, self.val_data = train_test_split(
            self.dataset,
            train_size=self.frac_train,
            random_state=self.seed,
        )
        # self.val_data = self.val_data.dropna()
        self.check_data(self.train_data, "Train")
        self.check_data(self.val_data, "Val")
    @staticmethod
    def check_data(df: pd.DataFrame, prefix: str):
        if LOGGER.isEnabledFor(logging.DEBUG):
            for modality in df.columns:
                try:
                    data = np.array(df[modality].to_list())
                    LOGGER.debug(f"{prefix} data {modality} shape={data.shape}")
                    LOGGER.debug(f"NaNs in {modality} {np.isnan(data).sum()}")
                except (ValueError, TypeError):
                    LOGGER.debug(f"Data {modality} is not a numpy array. Skipping shape check.")
                    continue

    def get_modality_pairs(
        self,
    ) -> tuple[list[str], list[tuple[str, ...]]]:
        modalities = self.modalities.copy()
        if self.central_modality in modalities:
            modalities.remove(self.central_modality)
        return modalities, [(self.central_modality, modality) for modality in modalities]

    def build_dataloader(
        self,
        data: pd.DataFrame,
        shuffle: bool,
        drop_last: bool,
        combined_loader_mode: _LITERAL_SUPPORTED_MODES,
    ) -> tuple[dict[str, list[str]], CombinedLoader]:
        modalities, modality_pairs = self.get_modality_pairs()

        if self.trainer is None:
            raise ValueError("Trainer is not set. Please set the trainer before calling build_dataloader.")

        rank_and_num_replicas = getattr(
            self.trainer.strategy,
            "distributed_sampler_kwargs",
            {"rank": 0, "num_replicas": 1},
        )  # assume non-distributed if not present

        distributed_sampler_kwargs = {
            "shuffle": shuffle,
            "drop_last": drop_last,
            "seed": self.seed,
            **rank_and_num_replicas,
        }

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": drop_last,
            **self.dloader_kwargs,
        }
        if combined_loader_mode == "sequential":
            dataloader_kwargs["drop_last"] = False

        datasets = {}
        used_ids = {}
        central_modality_ids = []

        for modality, modality_pair in zip(modalities, modality_pairs, strict=False):
            sanitized_data = self.dataset_builder.get_modality_data(
                data,
                modality_pair,
            )
            dataset = self.dataset_builder.combine_datasets(
                self.dataset_builder.build_datasets(sanitized_data, modality_pair),
                dataloader_kwargs=dataloader_kwargs,
                distributed_sampler_kwargs=distributed_sampler_kwargs,
            )

            datasets[modality] = dataset

            used_ids[modality] = sanitized_data[self.id_column].to_list()
            central_modality_ids.extend(sanitized_data[self.id_column].to_list())

        used_ids[self.central_modality] = central_modality_ids

        return used_ids, CombinedLoader(
            datasets,
            mode=combined_loader_mode,
        )

    def train_dataloader(self):
        # IDs are not correctly ordered if we shuffle the data
        if self.train_data is None:
            raise ValueError("No training data available. Please check the setup method.")

        _, dloader = self.build_dataloader(
            self.train_data,
            drop_last=self.drop_last,
            shuffle=True,
            combined_loader_mode=self.combined_loader_mode,  # type: ignore
        )
        return dloader

    def val_dataloader(self):
        ids, dloader = self.build_dataloader(
            self.val_data,
            shuffle=False,
            drop_last=False,
            combined_loader_mode=self.combined_loader_mode,  # type: ignore
        )
        self.validation_ids = ids
        return dloader

    def predict_dataloader(self):
        ids, dloader = self.build_dataloader(
            self.val_data,
            shuffle=False,
            drop_last=False,
            combined_loader_mode="sequential",
        )
        self.prediction_ids = ids
        return dloader


def get_mock_data(
    central_modality: str,
    modalities: list[str],
    id_column: str = "id",
    num_features: int = 2,
    length: int = 20,
):
    def get_array(length: int, num_features: int):
        array = np.tile(np.arange(length).reshape(-1, 1), (1, num_features)).astype(np.float32)
        if num_features == 1:
            return array.flatten()
        return array

    data = {central_modality: get_array(length, num_features).tolist()}
    data[id_column] = [f"id_{i}" for i in range(length)]

    lengths = {modality: np.random.randint(1, length) for modality in modalities}

    for modality in modalities:
        if modality == central_modality:
            continue
        array = get_array(lengths[modality], num_features)
        num_nans = np.random.randint(1, max(int(lengths[modality] * 0.2), 2))

        if num_nans >= lengths[modality]:
            num_nans = lengths[modality] - 1
        nan_indices = np.random.choice(np.arange(lengths[modality]), size=num_nans, replace=False)

        data_ = array.tolist() + [None] * (length - len(array))
        for idx in nan_indices:
            data_[idx] = None
        data[modality] = data_

    mock_data = pd.DataFrame(data)
    return update_mock_data(mock_data)


def update_mock_data(mock_data: pd.DataFrame) -> pd.DataFrame:
    if "dos" in mock_data.columns:
        mock_data["energies"] = mock_data["dos"].apply(lambda x: np.arange(len(x)) if x else None)  # type: ignore
        mock_data["efermi"] = 1.0

    if "crystal_structure" in mock_data.columns:
        num_nodes = 1
        num_node_features = 96
        num_edges = 2 * num_nodes
        num_edge_features = 41
        mock_data["node_feature"] = mock_data["crystal_structure"].index.values
        mock_data["node_feature"] = mock_data["node_feature"].apply(lambda x: x * np.ones((num_nodes, num_node_features)))
        mock_data["edge_index"] = mock_data["crystal_structure"].apply(
            lambda x: np.random.randint(0, num_nodes, size=(2, num_edges)) if x else None  # type: ignore
        )
        mock_data["edge_attribute"] = mock_data["crystal_structure"].apply(
            lambda x: np.random.rand(num_edges, num_edge_features).astype(np.float32) if x else None  # type: ignore
        )

    return mock_data
