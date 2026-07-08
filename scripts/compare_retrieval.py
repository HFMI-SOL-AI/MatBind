from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    training_dir = Path("/p/project1/solai/oestreicher1/repos/MatBind/outputs/les_models")
    subsets = ["train_subset", "val"]

    paths = get_retrieval_metrics_subsets_paths(training_dir=training_dir, subsets=subsets)
    dfs = [get_retrieval_metrics(path) for path in paths]
    combined_df = pd.concat(dfs, ignore_index=True)

    plot_val_train_comparison(df=combined_df, save_path=training_dir / "retrieval_results")


def get_retrieval_metrics_subsets_paths(
    training_dir: Path,
    subsets: list[str],
) -> list[Path]:
    return [training_dir / "retrieval_results" / subset / "retrieval_metrics.csv" for subset in subsets]


def get_retrieval_metrics(path_to_csv: Path):
    df = pd.read_csv(path_to_csv)
    df["source"] = path_to_csv.parent.name
    return df.rename(columns={"Unnamed: 0": "metric"})


def plot_val_train_comparison(df: pd.DataFrame, save_path: Path):
    metrics = df["metric"].unique()
    modality_pairs = df.drop(columns=["metric", "source"]).columns.tolist()

    for metric in metrics:
        subset = df[df["metric"] == metric]

        # Create bar positions
        x = range(len(modality_pairs))
        width = 0.35

        plt.figure(figsize=(14, 6))

        # Plot train and val bars
        plt.bar(
            [p - width / 2 for p in x],
            subset[subset["source"] == "train_subset"][modality_pairs].iloc[0],
            width,
            label="train_subset",
        )
        plt.bar([p + width / 2 for p in x], subset[subset["source"] == "val"][modality_pairs].iloc[0], width, label="val")

        plt.xticks(x, modality_pairs, rotation=45, ha="right")
        plt.title(f"Comparison for {metric}")
        plt.ylabel("Value")
        plt.xlabel("Modality Pair")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path / f"{metric}_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()
