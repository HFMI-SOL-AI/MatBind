from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

import polars as pl
from pymatgen.core import Structure
from tqdm import tqdm

from matbind.data.datasets.crystal_structure.core import GaussianDistanceCalculator, create_graph_data

RADIUS = 8.0
STEP = 0.2
MAX_NUM_NBR = 12
NUM_CPUS = 8  # None uses all available CPUs, use carfully since it can lead to high memory usage


def process_structures_in_parallel(
    structures: list[dict[str, Any] | Structure],
    max_num_nbr: int,
    radius: float,
    calculators: list[GaussianDistanceCalculator],
    num_cpus: int | None = None,
) -> list[tuple[list[list[float]], list[list[int]], list[list[float]]]]:
    with ProcessPoolExecutor(max_workers=num_cpus) as pool:
        return list(
            tqdm(
                pool.map(create_graph_data, structures, repeat(max_num_nbr), repeat(radius), repeat(calculators)),
                total=len(structures),
                desc="Generating graph data from structures",
            )
        )


def preprocessing(dataset: pl.DataFrame) -> pl.DataFrame:
    calculators = [GaussianDistanceCalculator(dmin=0, dmax=RADIUS, step=STEP)]
    computed_graphs = process_structures_in_parallel(
        structures=dataset["crystal_structure"].to_list(),
        radius=RADIUS,
        max_num_nbr=MAX_NUM_NBR,
        calculators=calculators,
        num_cpus=NUM_CPUS,
    )

    # Convert list of tuples to tuple of lists
    node_features, edge_index, edge_attr = tuple(map(list, zip(*computed_graphs, strict=False)))

    return dataset.with_columns(
        pl.Series(
            name="node_features",
            values=node_features,
            strict=False,
        ),
        pl.Series(
            name="edge_index",
            values=edge_index,
            strict=False,
        ),
        pl.Series(
            name="edge_attr",
            values=edge_attr,
            strict=False,
        ),
    )


def main():
    dataset_source = Path("/p/project1/solai/datasets/materials_project_cubic_only/materials_project_cubic_only_deviated.parquet")
    destination = (
        dataset_source.parent / f"{dataset_source.stem}_processed.parquet"
    )  # if you want to keep the original file, otherwise just use dataset_source
    dataset = pl.read_parquet(dataset_source)  # adjust reading function to your file format

    print("Initial dataset:")
    print(dataset)

    if "crystal_structure" not in dataset.columns:
        raise ValueError("The dataset does not contain 'crystal_structure' column.")

    dataset = preprocessing(dataset)
    print("Processed dataset:")
    print(dataset)
    dataset.write_parquet(destination)  # adjust writing function to your file format
    read = pl.read_parquet(destination)
    print("Read dataset:")
    print(read)


if __name__ == "__main__":
    main()
