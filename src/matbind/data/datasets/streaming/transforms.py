import numpy as np
import pandas as pd
import torch

from matbind.data.datasets.crystal_structure.dataset import CrystalGraphData


class RenameColumns:
    def __init__(self, columns_map: dict[str, str]):
        self.columns_map = columns_map

    def __call__(self, row: pd.Series) -> pd.Series:
        return row.rename(self.columns_map)


class CrystalGraphFromColumns:
    def __init__(
        self,
        new_column: str = "crystal_structure",
    ):
        self.new_column = new_column

    def __call__(self, row: pd.Series) -> pd.Series:
        x = row["node_features"]
        edge_index: torch.Tensor = row["edge_index"]
        edge_index = edge_index.long()
        edge_attr = row["edge_attr"]

        row[self.new_column] = CrystalGraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return row.drop(labels=["node_features", "edge_index", "edge_attr"])


class ToTensor:
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def __call__(self, row: pd.Series) -> pd.Series:
        columns = self.columns or row.index
        for col in columns:
            value = row[col]
            if value is None:
                continue

            if isinstance(value, np.ndarray) and value.dtype == np.object_:
                # check if all elements are numpy arrays of same shape
                if all(isinstance(v, np.ndarray) for v in value):
                    value = np.stack(value)

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            row[col] = torch.atleast_1d(value)

        return row


class To2DTensor:
    """Turns a column (or columns) that already contains a tensor into a 2D tensor."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def __call__(self, row: pd.Series) -> pd.Series:
        for col in self.columns:
            value = row[col]
            if value is None:
                continue

            row[col] = torch.atleast_2d(value)

        return row


class AlignToCentralModality:
    """
    Aligns all other modalities in a row to the given central modality.

    Each non-central modality becomes a dictionary containing:
      {
        modality_name: tensor(value_of_modality),
        central_modality: tensor(value_of_central_modality)
      }

    Missing values (None) are skipped.

    Returns:
        pd.Series where each key is a modality name (except the central one),
        and each value is a dict of aligned tensors.
    """

    def __init__(
        self,
        central_modality: str,
        modalities: list[str] | None = None,
    ):
        self.central_modality = central_modality
        self.modalities = modalities.copy()
        if self.modalities is not None and self.central_modality in self.modalities:
            self.modalities.remove(self.central_modality)

    def __call__(self, row: pd.Series) -> pd.Series:
        modalities = self.modalities or [m for m in row.index if m != self.central_modality]
        result = {}

        central_value = row[self.central_modality]
        if central_value is None:
            return pd.Series(dtype=object)

        for modality in modalities:
            value = row[modality]
            if value is None:
                continue

            result[modality] = {
                self.central_modality: central_value,
                modality: value,
            }

        return pd.Series(result, dtype=object)


class TokenizeText:
    def __init__(
        self,
        tokenizer,
        text_column: str = "text",
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.text_column = text_column

        self.max_length = max_length

    def __call__(self, row: pd.Series) -> pd.Series:
        text = row[self.text_column]

        if text is None:
            return row

        encoding: dict[str, torch.Tensor] = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        tokens = (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
        )

        row[self.text_column] = tokens
        return row


class ToDict:
    def __init__(self):
        pass

    def __call__(self, row: pd.Series) -> dict:
        return row.to_dict()
