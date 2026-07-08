from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------
# 1. Load all CSV files
# ----------------------------


def get_csv_files():
    data_dir = Path(__file__).parent / "data"
    return sorted(data_dir.glob("*.csv"))


def get_results_dir(mk_dirs: bool) -> Path:
    results_dir = Path(__file__).parent / "results"
    if mk_dirs:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def main():
    files = get_csv_files()
    results_dir = get_results_dir(mk_dirs=True)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["central_modality"] = f.stem  # label the CSV file that data came from
        dfs.append(df)

    data: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    data = data.rename(columns={"Unnamed: 0": "metric"})
    print("Combined Data:")
    print(data)

    # ----------------------------
    # 2. Long-format transformation
    # ----------------------------
    # Melt wide columns (Recall@1, Recall@5 with pairwise metrics)
    long = data.melt(id_vars=["central_modality", "metric"], var_name="pair", value_name="value")

    # Drop rows where pair is the metric column itself
    long = long[long["pair"] != "metric"]

    print("\nLong-format Data:")
    print(long)

    # ----------------------------
    # 3. Create a heatmap for each metric
    # ----------------------------
    # Loop through each metric (Recall@1, Recall@5)
    for metric in long["metric"].unique():
        save_path = results_dir / f"heatmap_{metric.replace('@', '_at_')}.png"
        subset = long[long["metric"] == metric]

        pivot = subset.pivot_table(index="pair", columns="central_modality", values="value")

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Heatmap for {metric}")
        plt.xlabel("Central Modality")
        plt.ylabel("Pair Relationship")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # ----------------------------
    # 4. Grouped bar chart for both metrics
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.barplot(data=long, x="pair", y="value", hue="central_modality")
    plt.xticks(rotation=45, ha="right")
    plt.title("Grouped Bar Chart: All Metrics & CSV Files")
    plt.tight_layout()
    plt.savefig(results_dir / "grouped_bar_chart_all_metrics.png")
    plt.close()


if __name__ == "__main__":
    main()
