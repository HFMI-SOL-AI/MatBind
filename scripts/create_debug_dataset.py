from pathlib import Path

import polars as pl

from matbind.data.loading import MultiModalDatasetLoader


def pyarrow_serialize_structure_dict(sdict: dict):
    """Remove empty properties from a pymatgen Structure dictionary."""
    if sdict.get("properties") == {}:
        sdict.pop("properties")

    sites: list[dict] = sdict.get("sites", [])
    for site in sites:
        if site.get("properties") == {}:
            site.pop("properties")

    return sdict


def main():
    num_samples = 1000
    modalities = ["crystal_structure", "pxrd", "dos", "text"]
    data_loader = MultiModalDatasetLoader(
        modalities_to_load=modalities,
        merge_on="material_id",
        merge_how="inner",
    )
    data_dir = Path("/p/project1/solai/datasets/materials_project")

    df = data_loader.load_dataset(data_dir)
    df = df.dropna()
    df = df.sample(n=num_samples, random_state=42)
    df["crystal_structure"] = df["crystal_structure"].apply(pyarrow_serialize_structure_dict)

    save_dir = Path.cwd() / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pl.from_pandas(df)
    df.write_parquet(save_dir / f"materials_project_{num_samples}.parquet")


if __name__ == "__main__":
    # main()

    df = pl.read_parquet("data/materials_project_1000.parquet")
    df = df.to_pandas()

    densities = df["dos"].to_list()

    print(densities[0].shape)
