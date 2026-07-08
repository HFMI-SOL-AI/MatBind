import datetime
import pickle as pkl
from collections.abc import Callable
from pathlib import Path

import hydra
import pandas as pd
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from matbind.data.datamodules.matbind import MatBindDataModule
from matbind.model.lightning_module import MatBindModule
from matbind.metrics.retrieval_faiss import faiss_retrieval
from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)

load_dotenv()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

RETRIEVAL_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")


class ModalityEmbeddings:
    """Stores embeddings for a modality with their corresponding material IDs and indices."""

    def __init__(self, embeddings: torch.Tensor, material_ids: list[str], indices: list[int]):
        self.embeddings = embeddings
        self.material_ids = material_ids
        self.indices = indices

    def __len__(self):
        return len(self.material_ids)


def embed_modality_batch(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    modality: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Embed all samples in a dataset for a specific modality.

    Args:
        model: The MatBind model
        dataset: PyTorch dataset for the modality
        modality: Name of the modality
        batch_size: Batch size for embedding
        device: Device to use

    Returns:
        Tensor of embeddings (N, D) where N is the number of samples
    """
    dataloader_class = GeometricDataLoader if isinstance(dataset, GeometricDataset) else DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {modality}", leave=False):
            if isinstance(batch, dict):
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch_data = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif hasattr(batch, "to"):
                batch_data = batch.to(device)
            else:
                batch_data = batch

            embeddings = model.model.encode(modality, batch_data, with_projection=True)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def embed_all_modalities(
    model: MatBindModule,
    data: pd.DataFrame,
    modalities: list[str],
    batch_size: int,
    modality_to_dataset: dict[str, Callable],
    device: str,
) -> dict[str, ModalityEmbeddings]:
    """
    Embed each sample exactly once for each modality.

    Args:
        model: The MatBind model (already on device and in eval mode)
        data: DataFrame with samples
        modalities: list of modality names
        batch_size: Batch size for embedding
        modality_to_dataset: dict mapping modality names to dataset constructors
        device: Device to use

    Returns:
        dictionary mapping modality name to ModalityEmbeddings
    """

    LOGGER.info(f"Total samples to embed: {len(data)}")

    embeddings_dict = {}

    for modality in modalities:
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"Processing modality: {modality}")
        LOGGER.info(f"{'=' * 60}")

        # Get samples where this modality is available
        available_mask = data[modality].notna()
        available_indices = available_mask[available_mask].index.tolist()
        n_available = len(available_indices)

        LOGGER.info(f"Samples with {modality}: {n_available}/{len(data)} ({100 * n_available / len(data):.1f}%)")

        if n_available == 0:
            LOGGER.warning(f"No samples available for {modality}, skipping")
            embeddings_dict[modality] = ModalityEmbeddings(
                embeddings=torch.empty(0),
                material_ids=[],
                indices=[],
            )
            continue

        # Get the data for this modality
        modality_data = data.loc[available_indices]

        # Build dataset for this modality
        dataset_factory = modality_to_dataset.get(modality)
        if dataset_factory is None:
            raise ValueError(f"No dataset factory found for modality {modality}")

        dataset = dataset_factory(modality_data)
        LOGGER.info(f"Created dataset with {len(dataset)} samples")

        # Embed the dataset
        embeddings = embed_modality_batch(
            model=model,
            dataset=dataset,
            modality=modality,
            batch_size=batch_size,
            device=device,
        )

        # Store embeddings with metadata
        material_ids = modality_data["material_id"].tolist()

        embeddings_dict[modality] = ModalityEmbeddings(
            embeddings=embeddings,
            material_ids=material_ids,
            indices=available_indices,
        )

        LOGGER.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    return embeddings_dict


def align_embeddings_by_material_id(
    original_embeddings: ModalityEmbeddings,
    noised_embeddings: ModalityEmbeddings,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Align original and noised embeddings by matching material IDs.

    Returns:
        - original embeddings (N, D)
        - noised embeddings (N, D)
        - list of aligned material IDs
    """
    # Create mapping from material_id to position
    orig_id_to_pos = {mid: i for i, mid in enumerate(original_embeddings.material_ids)}
    noised_id_to_pos = {mid: i for i, mid in enumerate(noised_embeddings.material_ids)}

    # Find common material IDs
    orig_ids_set = set(original_embeddings.material_ids)
    noised_ids_set = set(noised_embeddings.material_ids)
    common_ids = sorted(orig_ids_set & noised_ids_set)

    LOGGER.info(f"Original embeddings: {len(original_embeddings.material_ids)} samples")
    LOGGER.info(f"Noised embeddings: {len(noised_embeddings.material_ids)} samples")
    LOGGER.info(f"Common material IDs: {len(common_ids)}")

    if len(common_ids) == 0:
        raise ValueError("No common material IDs between original and noised embeddings")

    # Get positions for common IDs
    orig_positions = [orig_id_to_pos[mid] for mid in common_ids]
    noised_positions = [noised_id_to_pos[mid] for mid in common_ids]

    # Extract aligned embeddings
    aligned_orig = original_embeddings.embeddings[orig_positions]
    aligned_noised = noised_embeddings.embeddings[noised_positions]

    return aligned_orig, aligned_noised, common_ids


def load_model_from_checkpoint(
    training_config: DictConfig,
    ckpt_path: Path,
    device: str,
) -> MatBindModule:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model: MatBindModule = hydra.utils.instantiate(
        training_config.model,
        world_size=1,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


def set_model_to_eval_float32(model: MatBindModule, device: str) -> MatBindModule:
    """Convert model to float32 for inference."""
    LOGGER.info(f"Converting model to float32 and moving to device {device} for inference")

    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    for name, module in model.named_modules():
        if hasattr(module, "dtype") and module.dtype is not None:
            if module.dtype == torch.bfloat16:
                LOGGER.info(f"  Changing {name}.dtype from bfloat16 to float32")
                module.dtype = torch.float32

    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    LOGGER.info("Model ready for inference")
    return model


def retrieve_noised_from_original(
    original_embeddings: torch.Tensor,
    noised_embeddings: torch.Tensor,
    material_ids: list[str],
    top_k: list[int],
    noise_level: str | float,
) -> pd.DataFrame:
    """
    Perform retrieval: use noised embeddings as queries, original as gallery.

    Args:
        original_embeddings: Gallery embeddings (N, D)
        noised_embeddings: Query embeddings (N, D) - same materials as original
        material_ids: Common material IDs
        top_k: List of k values for recall@k
        noise_level: Label for this noise level

    Returns:
        DataFrame with retrieval metrics
    """
    LOGGER.info(f"\nRetrieving noised data from original (gallery) embeddings")
    LOGGER.info(f"Query embeddings: {noised_embeddings.shape}")
    LOGGER.info(f"Gallery embeddings: {original_embeddings.shape}")

    # Prepare indices DataFrame compatible with faiss_retrieval
    indices_df = pd.DataFrame(
        {
            "material_id": material_ids,
            "original": [1.0] * len(material_ids),
            "noised": [1.0] * len(material_ids),
        }
    )

    embeddings_dict = {
        "original": original_embeddings,
        "noised": noised_embeddings,
    }

    try:
        retrieval_metrics = faiss_retrieval(
            embeddings=embeddings_dict,
            indices=indices_df,
            all_modalities=["original", "noised"],
            top_k=top_k,
            use_gpu=False,
        )
    except Exception as exc:
        LOGGER.warning(f"FAISS retrieval failed: {exc}")
        return pd.DataFrame()

    metrics_key = "original_noised"
    metrics = retrieval_metrics.get(metrics_key, {})

    if not metrics:
        LOGGER.warning(f"No metrics returned for {metrics_key}")
        return pd.DataFrame()

    metrics_with_noise = {"noise_level": noise_level}
    metrics_with_noise.update({k.lower(): v for k, v in metrics.items()})

    LOGGER.info(f"Results for noise level {noise_level}:")
    LOGGER.info(f"  Recall metrics: {metrics_with_noise}")

    return pd.DataFrame([metrics_with_noise])


def main_evaluate_noise_robustness(
    training_config: DictConfig,
    ckpt_path: Path,
    results_save_location: Path,
    retrieval_batch_size: int,
    modality: str,
    noise_levels: list[str | float],
    top_k: list[int],
):
    """
    Evaluate retrieval performance between original and noised data.

    Args:
        training_config: Hydra training configuration
        ckpt_path: Path to model checkpoint
        results_save_location: Where to save results
        retrieval_batch_size: Batch size for embedding
        modality: Which modality to evaluate (e.g., "pxrd" or "dos")
        noise_levels: List of noise levels to evaluate
        top_k: List of k values for recall@k metrics
        noised_data_dir: Directory containing noised datasets (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")

    LOGGER.info(f"Loading model from checkpoint: {ckpt_path}")
    model = load_model_from_checkpoint(training_config, ckpt_path, device)
    model = set_model_to_eval_float32(model, device)

    # Load original data
    datamodule: MatBindDataModule = hydra.utils.instantiate(training_config.data)
    datamodule.setup()
    original_data = datamodule.val_data.reset_index(drop=True)

    # Embed original data
    LOGGER.info(f"\n{'#' * 60}\nEmbedding original data for {modality}\n{'#' * 60}")
    original_embeddings_dict = embed_all_modalities(
        model=model,
        data=original_data,
        modalities=[modality],
        batch_size=retrieval_batch_size,
        modality_to_dataset=datamodule.dataset_builder.modality_to_dataset,
        device=device,
    )

    results_save_location.mkdir(parents=True, exist_ok=True)

    # Evaluate noise robustness for the selected modality
    all_results = []

    LOGGER.info(f"\n{'#' * 60}\nEvaluating {modality}\n{'#' * 60}")

    if modality not in original_embeddings_dict:
        LOGGER.error(f"Modality {modality} not in original embeddings")
        return

    original_emb = original_embeddings_dict[modality]

    if len(original_emb) == 0:
        LOGGER.error(f"No original embeddings for {modality}")
        return

    # Create output directory for this modality
    modality_save_dir = results_save_location / modality
    modality_save_dir.mkdir(parents=True, exist_ok=True)

    # For each noise level
    for noise_level in noise_levels:
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"Processing noise level: {noise_level}")
        LOGGER.info(f"{'=' * 60}")

        try:
            # Create noised dataset with the specified noise level
            LOGGER.info(f"Creating noised dataset for {modality} with noise level {noise_level}")

            try:
                # Create a wrapped dataset factory that applies noise parameters
                base_factory = datamodule.dataset_builder.modality_to_dataset[modality]

                def noised_dataset_factory(data):
                    """Factory that creates noised datasets with the specified noise level."""
                    try:
                        return base_factory(data, noised=True, scale=noise_level)
                    except TypeError:
                        # Fallback if dataset doesn't support noise parameters
                        LOGGER.warning(f"Dataset for {modality} doesn't support noise parameters, creating without noise")
                        return base_factory(data)

                # Embed noised data using the wrapped factory
                noised_embeddings_dict = embed_all_modalities(
                    model=model,
                    data=original_data,
                    modalities=[modality],
                    batch_size=retrieval_batch_size,
                    modality_to_dataset={modality: noised_dataset_factory},
                    device=device,
                )

                noised_emb = noised_embeddings_dict[modality]

            except Exception as e:
                LOGGER.warning(f"Could not create noised data for {modality} at noise level {noise_level}: {e}")
                continue

            # Align embeddings by material ID
            aligned_orig, aligned_noised, common_ids = align_embeddings_by_material_id(
                original_emb,
                noised_emb,
            )

            # Evaluate retrieval
            metrics_df = retrieve_noised_from_original(
                original_embeddings=aligned_orig,
                noised_embeddings=aligned_noised,
                material_ids=common_ids,
                top_k=top_k,
                noise_level=noise_level,
            )

            if not metrics_df.empty:
                all_results.append(metrics_df)

                # Save per-noise-level results
                metrics_path = modality_save_dir / f"metrics_noise_{noise_level}.csv"
                metrics_df.to_csv(metrics_path, index=False)
                LOGGER.info(f"Saved metrics to {metrics_path}")

        except Exception as e:
            LOGGER.error(f"Error processing noise level {noise_level} for {modality}: {e}")
            continue

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results_path = results_save_location / "noise_robustness_metrics.csv"
        combined_results.to_csv(combined_results_path, index=False)
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info("COMBINED RESULTS")
        LOGGER.info(f"{'=' * 60}")
        LOGGER.info(f"Saved combined metrics to {combined_results_path}")
        LOGGER.info(f"\n{combined_results.to_string()}")
    else:
        LOGGER.warning("No results generated")


@hydra.main(version_base="1.3", config_path="../configs", config_name="retrieval.yaml")
def main(config: DictConfig):
    """
    Main entry point for noise robustness evaluation.

    Usage:
        python experiments/retrieval_noise_robustness.py \
            'ckpt_path=/path/to/checkpoint.ckpt' \
            'results_save_location=/path/to/output' \
            'modality=pxrd' \
            'noise_levels=[0.001,0.01,0.1]' \
            'top_k=[1,5,10]' \
            'retrieval_batch_size=32'
    """
    training_config = config
    ckpt_path = Path(config.ckpt_path) if config.get("ckpt_path") else None
    results_save_location = Path(config.get("results_save_location", "outputs/noise_robustness"))

    # Get modality (single modality only)
    modality = config.get("modality", "pxrd")

    noise_levels = config.get("noise_levels", [0.0025, 0.01, 0.05, 0.1])

    top_k = config.get("top_k", [1, 5, 10])

    retrieval_batch_size = config.get("retrieval_batch_size", 32)

    LOGGER.info("Configuration:")
    LOGGER.info(f"  Checkpoint: {ckpt_path}")
    LOGGER.info(f"  Modality: {modality}")
    LOGGER.info(f"  Noise levels: {noise_levels}")
    LOGGER.info(f"  Top-k: {top_k}")
    LOGGER.info(f"  Batch size: {retrieval_batch_size}")

    main_evaluate_noise_robustness(
        training_config=training_config,
        ckpt_path=ckpt_path,
        results_save_location=results_save_location,
        retrieval_batch_size=retrieval_batch_size,
        modality=modality,
        noise_levels=noise_levels,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
