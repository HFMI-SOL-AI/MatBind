from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset as ConventionalDataset
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

Dataset = ConventionalDataset | GeometricDataset


class MultiModalDatasetBuilder:
    def __init__(
        self,
        modality_to_dataset: dict[str, Callable[[Iterable], Dataset]],
    ):
        self.modality_to_dataset = modality_to_dataset

    def get_modality_data(
        self,
        data: pd.DataFrame,
        modalities: tuple[str, ...],
    ) -> pd.DataFrame:
        self._check_data(data, modalities)
        data = data.dropna(subset=modalities)
        data = data.reset_index(drop=True)
        if len(data) == 0:
            raise ValueError(f"No data available for {modalities} after dropping NaNs.")
        return data

    def build_datasets(
        self,
        data: pd.DataFrame,
        modalities: tuple[str, ...],
    ) -> dict[str, Dataset]:
        data = self.get_modality_data(data, modalities)

        datasets = {}

        for modality in modalities:
            dataset_factory = self.modality_to_dataset.get(modality)
            if dataset_factory is None:
                raise ValueError(f"Dataset builder for modality {modality} not found.")
            datasets[modality] = dataset_factory(data)

        return datasets

    def combine_datasets(
        self,
        datasets: dict[str, Dataset],
        dataloader_kwargs: dict[str, Any],
        distributed_sampler_kwargs: dict[str, Any],
    ) -> CombinedLoader:
        self._check_equal_length(datasets)

        dataloaders = {}
        first_dataset, *_ = datasets.values()

        shared_distributed_sampler = DistributedSampler(
            first_dataset,
            **distributed_sampler_kwargs,
        )

        for modality, dataset in datasets.items():
            dataloader_factory = GeometricDataLoader if isinstance(dataset, GeometricDataset) else DataLoader

            dataloaders[modality] = dataloader_factory(
                dataset,  # type: ignore
                sampler=shared_distributed_sampler,
                **dataloader_kwargs,
            )

        return CombinedLoader(dataloaders)

    @staticmethod
    def _check_data(
        data: pd.DataFrame,
        modalities: tuple[str, ...],
    ):
        if not all(modality in data.columns for modality in modalities):
            raise ValueError(f"Not all modalities {modalities} are present in the data. Available columns: {data.columns}")

    @staticmethod
    def _check_equal_length(datasets: dict[str, Dataset]):
        lengths = [len(dataset) for dataset in datasets.values()]  # type: ignore
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Datasets have different lengths: {lengths}. All datasets must have the same length and have to be index aligned."
            )
