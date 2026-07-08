import random
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from pyarrow.parquet import ParquetFile
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from matbind.data.datasets.streaming.parquet import (
    StreamingParquetDataset,
)
from matbind.data.datasets.streaming.transformable import TransformDataset, TransformPipeline
from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


class StreamingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str,
        transform: TransformPipeline,
        columns_to_load: list[str] | None = None,
        modalities: list[str] | None = None,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        batch_size: int = 2048,
        drop_last: bool = True,
        buffer_size: int = 10000,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()

        data_dir = Path(data_dir)
        self.columns_to_load = columns_to_load
        self.transform = transform
        self.seed = seed
        self.modalities = modalities

        files = sorted(data_dir.glob("*.parquet"))
        self.report_num_samples(files)
        (
            self.train_files,
            self.validation_files,
            self.test_files,
        ) = self._split_files(
            files,
            frac_train,
            frac_val,
            frac_test,
            seed,
        )

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.num_workers = num_workers

    @staticmethod
    def report_num_samples(files: list[Path]):
        total = 0
        for file in files:
            pq_file = ParquetFile(file)
            num_samples = pq_file.metadata.num_rows
            LOGGER.debug(f"{num_samples} samples in  in {file.name}")
            total += num_samples

        LOGGER.info(f"Total samples found in dataset: {total}")

    def _split_files(
        self,
        files: list[Path],
        frac_train: float,
        frac_val: float,
        frac_test: float,
        seed: int,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        random.seed(seed)
        random.shuffle(files)
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
            "Fractions of training, validation, and test sets must sum to 1."
        )

        n = len(files)
        train_files = files[: int(frac_train * n)]
        validation_files = files[int(frac_train * n) : int((frac_train + frac_val) * n)]
        test_files = files[int((frac_train + frac_val) * n) :]
        return train_files, validation_files, test_files

    def get_rank_and_world_size(self) -> tuple[int, int]:
        if self.trainer is None:
            raise ValueError("Trainer is not set. Please set the trainer before calling build_dataloader.")

        rank_and_num_replicas = getattr(
            self.trainer.strategy,
            "distributed_sampler_kwargs",
            {"rank": 0, "num_replicas": 1},
        )
        return rank_and_num_replicas["rank"], rank_and_num_replicas["num_replicas"]

    def _make_loader(
        self,
        files: list[Path],
        shuffle: bool = False,
    ):
        rank, world_size = self.get_rank_and_world_size()

        dataset = TransformDataset(
            StreamingParquetDataset(
                parquet_files=files,
                columns=self.columns_to_load,
                seed=self.seed,
                buffer_size=self.buffer_size,
                rank=rank,
                world_size=world_size,
                shuffle=shuffle,
            ),
            transform=self.transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_stack_per_modality,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_files, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.validation_files, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_files, shuffle=False)

    def predict_dataloader(self):
        return self._make_loader(self.test_files, shuffle=False)


def collate_stack_per_modality(
    batch: list[dict[str, torch.Tensor | Data]],
) -> dict[str, dict[str, torch.Tensor | Data]]:
    """
    Collate a batch of dicts (from `align_to_central_modality`) into:
    { modality_name: { modality_name: Tensor, central_modality: Tensor }, ... }

    - Skips missing modalities (None or empty dicts)
    - Stacks tensors across the batch (assumes shapes are consistent)
    """
    collated: dict[str, dict[str, torch.Tensor | Data]] = {}

    batch = [b for b in batch if b is not None]

    # Collect all modality keys that appear in any sample
    all_modalities = set().union(*(b.keys() for b in batch))

    for modality in all_modalities:
        # Gather valid entries for this modality
        valid = [b[modality] for b in batch if isinstance(b, dict) and modality in b and b[modality] is not None]
        if not valid:
            continue

        collated_modality = {}
        for k in valid[0].keys():  # type: ignore # noqa: SIM118
            values = [v[k] for v in valid]  # type: ignore
            if all(isinstance(v, Data) for v in values):
                collated_modality[k] = Batch.from_data_list(values)  # type: ignore
            elif all(isinstance(v, tuple) for v in values):
                collated_modality[k] = tuple(torch.stack(items) for items in zip(*values, strict=True))  # type: ignore
            else:  # assume just regular tensors
                collated_modality[k] = torch.stack([v[k] for v in valid])  # type: ignore

        collated[modality] = collated_modality

    return collated
