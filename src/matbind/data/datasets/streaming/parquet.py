import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


class StreamingParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_files: list[Path],
        columns: list[str] | None = None,
        buffer_size: int = 10000,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ):
        self.files = parquet_files

        self.buffer_size = buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.columns = columns
        self.shuffle = shuffle

    def _yield_rows(self, file):
        pq_file = pq.ParquetFile(file)
        for batch in pq_file.iter_batches(
            batch_size=self.buffer_size * 2,
            columns=self.columns,
        ):
            df: pd.DataFrame = batch.to_pandas()
            for _, row in df.iterrows():
                yield row

    def __iter__(self):
        files = self.files[self.rank :: self.world_size]
        rng = random.Random(self.seed + self.rank)

        if self.shuffle:
            rng.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
            LOGGER.info(
                f"Dataloader worker {worker_info.id} on rank {self.rank}: {len(files)} files assigned: {[file.name for file in files]}"
            )

        buffer = []
        for file in files:
            for row in self._yield_rows(file):
                buffer.append(row)
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        rng.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()
