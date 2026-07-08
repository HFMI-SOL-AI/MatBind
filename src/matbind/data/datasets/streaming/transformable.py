from collections.abc import Callable

import pandas as pd
from torch.utils.data import IterableDataset

Transform = Callable[[pd.Series], pd.Series]


class TransformPipeline:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, row: pd.Series) -> pd.Series:
        for t in self.transforms:
            row = t(row)
        return row


class TransformDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for row in self.dataset:
            row = self.transform(row)
            yield row
