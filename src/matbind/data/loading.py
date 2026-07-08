from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd
import polars as pl
from pandas._typing import MergeHow

from matbind.utils.progress_bar import RankZeroOnlyTqdm

DATALOADING_HANDLERS: dict[str, Callable[[str | Path], pl.LazyFrame]] = {
    ".csv": pl.scan_csv,
    ".pickle": lambda path: pl.from_pandas(pd.read_pickle(path)).lazy(),
    ".pkl": lambda path: pl.from_pandas(pd.read_pickle(path)).lazy(),
    ".parquet": pl.scan_parquet,
    ".json": lambda path: pl.read_json(path).lazy(),
    ".jsonl": lambda path: pl.read_ndjson(path).lazy(),
}


DATALOADING_HANDLERS_PANDAS: dict[str, Callable[[str | Path], pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".pickle": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda path: pd.read_json(path, lines=True),
}


class DatasetLoader(Protocol):
    def load_dataset(self, path_to_dataset: Path) -> pd.DataFrame:
        raise NotImplementedError


class MultiModalDatasetLoader:
    def __init__(
        self,
        modalities_to_load: list[str] | None = None,
        merge_on: str = "material_id",
        merge_how: MergeHow = "inner",
    ):
        self.modalities_to_load = modalities_to_load or "all"
        self.merge_on = merge_on
        self.merge_how: MergeHow = merge_how

    def file_should_be_loaded(self, file: Path) -> bool:
        if file.suffix not in DATALOADING_HANDLERS:
            return False

        modality = file.stem
        return self.modalities_to_load == "all" or modality in self.modalities_to_load

    def load_dataset(self, path_to_dataset: Path) -> pd.DataFrame:
        if not path_to_dataset.is_dir():
            raise ValueError(f"Failed to load dataset. Provdided dataset path '{path_to_dataset.as_posix()}' is not a directory.")

        files = [path for path in path_to_dataset.iterdir() if path.is_file()]
        dataframes: list[pd.DataFrame] = []

        with RankZeroOnlyTqdm(files, desc="Loading dataset", unit="file") as pbar:
            for file in files:
                pbar.set_postfix_str(f"file = {file.name}")

                if not self.file_should_be_loaded(file):
                    pbar.update(1)
                    continue

                dataset = load_dataset_from_file(file)

                if self.merge_on in dataset.columns:
                    dataframes.append(dataset)

                pbar.update(1)

            if not dataframes:
                raise ValueError(
                    f"Failed to load dataset. Provided directory '{path_to_dataset.as_posix()}' does not contain a dataset."
                )
        dataset = dataframes[0]
        for df in dataframes[1:]:
            dataset = dataset.merge(df, on=self.merge_on, how=self.merge_how)
        return dataset


def load_dataset(path_to_dataset: Path, dataset_loader: DatasetLoader | None = None) -> pd.DataFrame:
    if path_to_dataset.is_file():
        return load_dataset_from_file(path_to_dataset)

    if path_to_dataset.is_dir():
        if dataset_loader is not None:
            return dataset_loader.load_dataset(path_to_dataset)
        # TODO: function to load dataset from directory without dataset loader (for example polars supports a parquet dataset stored as multiple files in a directory)
        raise ValueError(
            f"Failed to load dataset. Provided path '{path_to_dataset.as_posix()}' is a directory but no dataset loader was provided."
        )

    raise ValueError(f"Failed to load dataset. Provided path '{path_to_dataset.as_posix()}' is neither a file nor a directory.")


def load_dataset_from_file(path_to_dataset: Path) -> pd.DataFrame:
    data_format = path_to_dataset.suffix

    handler = DATALOADING_HANDLERS_PANDAS.get(data_format)

    if handler is None:
        raise ValueError(f"Error when trying to load the dataset. Format {data_format} not supported.")

    return handler(path_to_dataset)


if __name__ == "__main__":
    # Example usage
    dataset_path = Path("/p/project1/solai/datasets/materials_project")
    dataset_loader = MultiModalDatasetLoader(modalities_to_load=["pxrd", "text", "crystal_structure", "dos"], merge_how="outer")
    dataset = load_dataset(dataset_path, dataset_loader=dataset_loader)
    print(dataset)
    print(dataset.columns)
