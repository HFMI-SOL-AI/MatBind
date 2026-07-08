import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from plotreset import Styles


style = Styles("academic")
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Apply rc settings to seaborn as well so axis/legend text stay sharp.
sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)


def _remove_zero_label(value: float, _: int) -> str:
    """Formatter to hide the zero tick label while keeping spacing."""
    return "" if np.isclose(value, 0.0) else f"{value:.1f}"


def read_top_row(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    # take the first row as a Series
    s = df.iloc[0]
    s.index = s.index.astype(str)
    return s


def performance_over_k():
    selected_link = [
        "crystal_structure_text",
        "crystal_structure_dos",
        "crystal_structure_pxrd",
        "dos_text",
        "pxrd_text",
        "dos_pxrd",
    ]
    direct_link = ["crystal_structure_text", "crystal_structure_dos", "crystal_structure_pxrd"]
    csv_file = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260119_1203.csv"
    df = pd.read_csv(csv_file, index_col=0)
    df = df[selected_link]
    # prepare long-form dataframe for seaborn: x=index, y=cell count, hue=column
    df_reset = df.reset_index()
    x_col = df_reset.columns[0]
    df_melt = df_reset.melt(id_vars=x_col, var_name="Pairs", value_name="Recall")

    # try to coerce x to numeric if possible (helps correct x-ordering)
    try:
        df_melt[x_col] = pd.to_numeric(df_melt[x_col])
    except Exception:
        pass

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))

    # dash direct links for visual distinction
    dashes = {pair: (2, 2) if pair not in direct_link else "" for pair in df_melt["Pairs"].unique()}
    ax = sns.lineplot(data=df_melt, x=x_col, y="Recall", hue="Pairs", style="Pairs", dashes=dashes, marker="o")
    ax.set_xlabel("k (retrievals)")
    ax.set_ylabel("Recall")
    ax.set_title("Performance over K")
    plt.tight_layout()
    plt.savefig("figure/performance_over_k.pdf", bbox_inches="tight")


def retrieval_performace():
    crystal_central = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260113_1414.csv"

    # trained (direct) and emergent (zero-shot) metric lists
    direct_link = [
        "crystal_structure_text",
        "text_crystal_structure",
        "crystal_structure_dos",
        "dos_crystal_structure",
        "crystal_structure_pxrd",
        "pxrd_crystal_structure",
    ]
    emergent_link = ["dos_text", "text_dos", "pxrd_text", "text_pxrd", "dos_pxrd", "pxrd_dos"]

    s1 = read_top_row(crystal_central)
    # melt so each row is (Metric, Modality, Score)
    s1_dirct_melt = s1.reset_index().melt(id_vars=s1.reset_index().columns[0], var_name="Modality", value_name="Score")
    # keep only the direct-link metrics and order them according to `direct_link`
    s1_dirct_melt = s1_dirct_melt.rename(columns={s1_dirct_melt.columns[0]: "Metric"})
    s1_dirct_melt = s1_dirct_melt[s1_dirct_melt["Metric"].isin(direct_link)].copy()
    # preserve the requested order for the Metric categorical axis
    s1_dirct_melt["Metric"] = pd.Categorical(s1_dirct_melt["Metric"], categories=direct_link, ordered=True)
    s1_dirct_melt = s1_dirct_melt.sort_values("Metric")
    # s1_emergent = s1[emergent_link]
    # s1_emergent_melt = s1_emergent.reset_index().melt(id_vars=s1_emergent.reset_index().columns[0], var_name="Modality", value_name="Score")

    # Create two stacked horizontal subplots: Direct (top) and Emergent (bottom)
    df = s1.reset_index().rename(columns={s1.reset_index().columns[0]: "Metric"})
    df = df[df["Metric"].isin(direct_link + emergent_link)].copy()

    def pretty_name(metric: str) -> str:
        toks = metric.split("_")
        src = toks[:-1]
        tgt = toks[-1]

        def lab(token_list):
            if "pxrd" in token_list:
                return "pXRD"
            if "dos" in token_list:
                return "DOS"
            if "text" in token_list:
                return "Text"
            if "crystal" in token_list or "structure" in token_list:
                return "Crystal"
            return " ".join([t.capitalize() for t in token_list])

        return f"{lab(src)} → {lab([tgt])}"

    # build lists for direct and emergent in specified order
    direct_scores = [float(df[df["Metric"] == m].iloc[0, 1]) if not df[df["Metric"] == m].empty else np.nan for m in direct_link]
    direct_labels = [pretty_name(m) for m in direct_link]

    emergent_scores = [
        float(df[df["Metric"] == m].iloc[0, 1]) if not df[df["Metric"] == m].empty else np.nan for m in emergent_link
    ]
    emergent_labels = [pretty_name(m) for m in emergent_link]

    # plotting with seaborn
    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    n_direct = len(direct_scores)
    n_emergent = len(emergent_scores)
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [n_direct, n_emergent]}, figsize=(6, 6)
    )

    # color scheme: sample two distinct colors from viridis colormap
    direct_color = plt.cm.viridis(0.3)
    emergent_color = plt.cm.viridis(0.7)

    # top: direct - prepare data for seaborn
    direct_df = pd.DataFrame({"Metric": direct_labels, "Score": direct_scores})
    sns.barplot(data=direct_df, y="Metric", x="Score", ax=ax_top, color=direct_color, orient="h")
    ax_top.set_title("Direct (Trained)")
    ax_top.set_ylabel("")

    # bottom: emergent - prepare data for seaborn
    emergent_df = pd.DataFrame({"Metric": emergent_labels, "Score": emergent_scores})
    sns.barplot(data=emergent_df, y="Metric", x="Score", ax=ax_bot, color=emergent_color, orient="h")
    ax_bot.set_title("Emergent (Zero-Shot)")
    ax_bot.set_ylabel("")

    # shared x-axis formatting
    ax_bot.set_xlim(0, 1.0)
    ax_bot.set_xticks(np.linspace(0, 1.0, 11))
    ax_bot.set_xlabel("Recall")

    plt.tight_layout()
    plt.savefig("figure/merged_retrieval.pdf", bbox_inches="tight")


def central_modality_compare():
    crystal_central = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260113_1414.csv"
    # file2 = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260114_2019.csv" # batch norm
    file2 = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260113_1334.csv"  # text central
    # file2 = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260114_1625.csv" # dos central
    label1, label2 = "crystal structure", "text"

    s1 = read_top_row(crystal_central).rename(label1)
    s2 = read_top_row(file2).rename(label2)

    combined = pd.concat([s1, s2], axis=1)

    # melt for seaborn grouped barplot
    df_melt = combined.reset_index().melt(id_vars=combined.reset_index().columns[0], var_name="Modality", value_name="Score")
    df_melt = df_melt.rename(columns={df_melt.columns[0]: "Metric"})

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Modality", palette="viridis")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Recall@1")
    ax.set_title(f"Comparison: {label1} vs {label2}")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figure/comparison_metrics_crystal_vs_text.pdf", bbox_inches="tight")


def impact_of_modality_combinations():
    crystal_pxrd_dos_text = "/p/project1/solai/yang21/MatBind/experiments/outputs/retrieval_metrics_20260113_1414.csv"
    crystal_pxrd = "/p/project1/solai/yang21/MatBind/experiments/crys_pxrd/val/retrieval_metrics.csv"
    crystal_pxrd_dos = "/p/project1/solai/yang21/MatBind/experiments/crys_pxrd_dos/val/retrieval_metrics.csv"

    s1 = read_top_row(crystal_pxrd_dos_text).rename("All Modalities")
    s2 = read_top_row(crystal_pxrd).rename("Crystal + pXRD")
    s3 = read_top_row(crystal_pxrd_dos).rename("Crystal + pXRD + DOS")

    combined = pd.concat([s1, s2, s3], axis=1)
    combined = combined.loc[["crystal_structure_pxrd", "pxrd_crystal_structure", "crystal_structure_dos", "dos_crystal_structure"]]

    # melt for seaborn grouped barplot
    df_melt = combined.reset_index().melt(id_vars=combined.reset_index().columns[0], var_name="Modality", value_name="Score")
    df_melt = df_melt.rename(columns={df_melt.columns[0]: "Metric"})

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Modality", palette="viridis")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Recall@1")
    ax.set_title(f"Comparison: All Modalities vs Crystal + pXRD")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figure/impact_of_modality_combinations.pdf", bbox_inches="tight")


def pxrd_noise_robustness():
    baseline = "/p/project1/solai/yang21/MatBind/experiments/pxrd_noise_robustness_results/val/retrieval_metrics.csv"
    noise_001875 = "/p/project1/solai/yang21/MatBind/experiments/pxrd_noise_robustness_0.001875/val/retrieval_metrics.csv"
    noise_000625 = "/p/project1/solai/yang21/MatBind/experiments/pxrd_noise_robustness_0.000625/val/retrieval_metrics.csv"
    noise_00125 = "/p/project1/solai/yang21/MatBind/experiments/pxrd_noise_robustness_0.00125/val/retrieval_metrics.csv"
    noise_0025 = "/p/project1/solai/yang21/MatBind/experiments/pxrd_noise_robustness_0.0025/val/retrieval_metrics.csv"

    # Load data from each noise level
    s_baseline = read_top_row(baseline).rename(0.0)
    s_000625 = read_top_row(noise_000625).rename(0.000625)
    s_00125 = read_top_row(noise_00125).rename(0.00125)
    s_001875 = read_top_row(noise_001875).rename(0.001875)
    s_0025 = read_top_row(noise_0025).rename(0.0025)

    # Combine all series
    combined = pd.concat([s_baseline, s_000625, s_00125, s_001875, s_0025], axis=1)

    # Sort columns by noise level
    combined = combined[sorted(combined.columns)]

    # Melt for seaborn lineplot
    df_melt = combined.reset_index().melt(id_vars=combined.reset_index().columns[0], var_name="Noise Level", value_name="Recall")
    df_melt = df_melt.rename(columns={df_melt.columns[0]: "Metric"})

    # Create plot
    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df_melt, x="Noise Level", y="Recall", hue="Metric", marker="o")
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Recall@1")
    ax.set_title("pXRD Noise Robustness")
    plt.tight_layout()
    plt.savefig("figure/noise_robustness.pdf", bbox_inches="tight")


def plot_noised_pxrd():
    import pandas as pd

    pxrd_dataset = pd.read_parquet("/p/project1/solai/datasets/materials_project/pxrd.parquet")
    pxrd = pxrd_dataset.iloc[0]["pxrd"]
    noise_levels = [0.0, 0.000625, 0.001875, 0.00125, 0.0025]
    noisy_pxrd_list = []
    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, size=pxrd.shape)
        noisy_pxrd = np.clip(pxrd + noise, 0, 1)
        noisy_pxrd_list.append(noisy_pxrd)

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)

    # Create subplots: 1 row, 5 columns for 5 noise levels
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)

    for idx, (sigma, noisy_pxrd) in enumerate(zip(noise_levels, noisy_pxrd_list)):
        ax = axes.flat[idx]
        ax.plot(noisy_pxrd)
        ax.set_xlabel("2θ")
        if idx == 0:
            ax.set_ylabel("Intensity")
        ax.set_title(f"σ={sigma}")

    plt.tight_layout()
    plt.savefig("figure/noised_pxrd.pdf", bbox_inches="tight")


def plot_noised_dos():
    import pandas as pd

    dos_dataset = pd.read_parquet("/p/project1/solai/datasets/materials_project/dos.parquet")
    dos = dos_dataset.iloc[1]["dos"]
    energy = dos_dataset.iloc[1]["energies"]
    efermi = dos_dataset.iloc[1]["efermi"]
    relative_energy = energy - efermi

    noise_levels = [0.0, 0.0025, 0.05, 0.01, 0.1]
    noisy_dos_list = []
    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, size=dos.shape)
        noisy_dos = dos + noise
        noisy_dos_list.append(noisy_dos)

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)

    # Create subplots: 1 row, 5 columns for 5 noise levels
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey=True)

    for idx, (sigma, noisy_dos) in enumerate(zip(noise_levels, noisy_dos_list)):
        ax = axes.flat[idx]
        ax.plot(relative_energy, noisy_dos)
        ax.set_xlabel("Relative Energy (eV)")
        if idx == 0:
            ax.set_ylabel("DOS")
        ax.set_title(f"σ={sigma}")

    plt.tight_layout()
    plt.savefig("figure/noised_dos.pdf", bbox_inches="tight")


def dos_noise_robustness():
    baseline = "/p/project1/solai/yang21/MatBind/experiments/dos_noise_robustness/val/retrieval_metrics.csv"
    noise_0025 = "/p/project1/solai/yang21/MatBind/experiments/dos_noise_robustness_0.0025/val/retrieval_metrics.csv"
    noise_01 = "/p/project1/solai/yang21/MatBind/experiments/dos_noise_robustness_0.01/val/retrieval_metrics.csv"
    noise_05 = "/p/project1/solai/yang21/MatBind/experiments/dos_noise_robustness_0.05/val/retrieval_metrics.csv"
    noise_1 = "/p/project1/solai/yang21/MatBind/experiments/dos_noise_robustness_0.1/val/retrieval_metrics.csv"

    # Load data from each noise level
    s_baseline = read_top_row(baseline).rename(0.0)
    s_0025 = read_top_row(noise_0025).rename(0.0025)
    s_01 = read_top_row(noise_01).rename(0.01)
    s_05 = read_top_row(noise_05).rename(0.05)
    s_1 = read_top_row(noise_1).rename(0.1)

    # Combine all series
    combined = pd.concat([s_baseline, s_0025, s_01, s_05, s_1], axis=1)

    # Filter to only metrics between DOS and crystal_structure
    dos_crystal_metrics = ["dos_crystal_structure", "crystal_structure_dos"]
    combined = combined.loc[combined.index.isin(dos_crystal_metrics)]

    # Sort columns by noise level
    combined = combined[sorted(combined.columns)]

    # Melt for seaborn lineplot
    df_melt = combined.reset_index().melt(
        id_vars=combined.reset_index().columns[0], var_name="Noise Level", value_name="Recall@1"
    )
    df_melt = df_melt.rename(columns={df_melt.columns[0]: "Metric"})

    # Create plot
    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df_melt, x="Noise Level", y="Recall@1", hue="Metric", marker="o")
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Recall@1")
    ax.set_title("DOS Noise Robustness")
    plt.tight_layout()
    plt.savefig("figure/dos_noise_robustness.pdf", bbox_inches="tight")


def crystal_system_classification():
    x = ["all_four_modalities_text", "all_four_modalities_crystal", "crystal_pxrd_crystal","crystal_pxrd_pxrd","random_guess"]
    y = [0.625, 0.54, 0.5399, 0.54303, 1 / 7]

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=x, y=y)
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Linear probe for Crystal System Classification")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figure/crystal_system_classification.pdf", bbox_inches="tight")



def crystal_system_classification_one_layer():
    x = ["all_four_modalities_crystal", "all_four_modalities_text", "pretrained_matbert","text_scratch","random_guess"]
    y = [0.3975, 0.7125, 0.8319, 0.7006, 1 / 7]

    sns.set_theme(context="paper", style="ticks", rc=plt.rcParams)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=x, y=y)
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Linear probe for Crystal System Classification")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figure/crystal_system_classification_one_layer.pdf", bbox_inches="tight")


if __name__ == "__main__":
    crystal_system_classification_one_layer()