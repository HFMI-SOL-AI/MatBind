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
import numpy as np

from matbind.data.datamodules.matbind import MatBindDataModule
from matbind.metrics.retrieval_faiss import faiss_retrieval, query, compute_retrieval_metrics, comput_rank
from matbind.model.lightning_module import MatBindModule
from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)

load_dotenv()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

RETRIEVAL_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")


class RetrievalConfig:
    training_dir: str
    ckpt_path: str | None
    results_save_location: str
    retrieval_batch_size: int
    top_k: list[int]


class ModalityEmbeddings:
    """Stores embeddings for a modality with their corresponding material IDs and indices."""

    def __init__(self, embeddings: torch.Tensor, material_ids: list[str], indices: list[int]):
        self.embeddings = embeddings
        self.material_ids = material_ids
        self.indices = indices

    def __len__(self):
        return len(self.material_ids)


def get_retrieval_datasets(datamodule: MatBindDataModule, config: DictConfig) -> list[tuple[str, pd.DataFrame]]:
    """Build retrieval datasets from config.

    Config options:
      - retrieval_splits: list[str] in {"val", "train_subset", "all"}
    """
    retrieval_splits = config.get("retrieval_splits", ["val"])
    datasets: list[tuple[str, pd.DataFrame]] = []

    for split in retrieval_splits:
        if split == "val":
            datasets.append(("val", prepare_data(datamodule, use_training_data_subset=False, all_data=False)))
        elif split == "train_subset":
            datasets.append(("train_subset", prepare_data(datamodule, use_training_data_subset=True, all_data=False)))
        elif split == "all":
            datasets.append(("all", prepare_data(datamodule, all_data=True)))
        else:
            LOGGER.warning(f"Unknown retrieval split '{split}', skipping")

    if not datasets:
        LOGGER.warning("No valid retrieval_splits provided, defaulting to ['val']")
        datasets = [("val", prepare_data(datamodule, use_training_data_subset=False, all_data=False))]

    return datasets


def run_chain_query_retrieval(
    embeddings_for_faiss: dict[str, torch.Tensor],
    aligned_df: pd.DataFrame,
    top_k: list[int],
    use_gpu: bool,
    config: DictConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Run configurable 2-stage chained retrieval and return metrics."""
    chain_metrics: dict[str, dict[str, float]] = {}
    chain_rank_results: dict[str, pd.DataFrame] = {}

    chain_cfg = config.get("chain_query", {})
    stage1_query_modality = chain_cfg.get("stage1_query_modality", "pxrd")
    stage1_target_modality = chain_cfg.get("stage1_target_modality", "crystal_structure")
    stage2_query_modality = chain_cfg.get("stage2_query_modality", "text")
    fusion_alpha = float(chain_cfg.get("fusion_alpha", 0.8))
    candidate_multiplier = int(chain_cfg.get("candidate_multiplier", 10))

    required_modalities = [stage1_query_modality, stage1_target_modality, stage2_query_modality]
    missing_modalities = [m for m in required_modalities if m not in aligned_df.columns or m not in embeddings_for_faiss]
    if missing_modalities:
        LOGGER.warning(f"Chain query skipped: missing modalities {missing_modalities}")
        return chain_metrics, chain_rank_results

    stage2_selected = aligned_df.loc[aligned_df[stage2_query_modality].notna(), "material_id"]
    stage1_target_selected = aligned_df.loc[aligned_df[stage1_target_modality].notna(), "material_id"]
    stage1_query_selected = aligned_df.loc[aligned_df[stage1_query_modality].notna(), "material_id"]

    stage2_series = stage2_selected if isinstance(stage2_selected, pd.Series) else pd.Series([stage2_selected])
    stage1_target_series = (
        stage1_target_selected if isinstance(stage1_target_selected, pd.Series) else pd.Series([stage1_target_selected])
    )
    stage1_query_series = (
        stage1_query_selected if isinstance(stage1_query_selected, pd.Series) else pd.Series([stage1_query_selected])
    )

    stage2_ids = [str(mid) for mid in stage2_series.tolist()]
    stage1_target_ids = [str(mid) for mid in stage1_target_series.tolist()]
    stage1_query_ids = [str(mid) for mid in stage1_query_series.tolist()]

    id_to_stage2_idx = {mid: i for i, mid in enumerate(stage2_ids)}
    id_to_stage1_query_idx = {mid: i for i, mid in enumerate(stage1_query_ids)}

    valid_query_ids = [mid for mid in stage1_target_ids if mid in id_to_stage2_idx and mid in id_to_stage1_query_idx]
    if len(valid_query_ids) == 0:
        LOGGER.warning("Chain query skipped: no overlapping samples across required modalities")
        return chain_metrics, chain_rank_results

    max_k = max(top_k)
    candidate_k = min(max_k * candidate_multiplier, len(stage1_target_ids))
    query_stage2_indices = [id_to_stage2_idx[mid] for mid in valid_query_ids]
    query_stage1_indices = [id_to_stage1_query_idx[mid] for mid in valid_query_ids]
    id_to_stage1_target_idx = {mid: i for i, mid in enumerate(stage1_target_ids)}
    gt_target_positions = np.asarray([id_to_stage1_target_idx[mid] for mid in valid_query_ids], dtype=np.int64)

    stage2_query_emb = embeddings_for_faiss[stage2_query_modality][torch.as_tensor(query_stage2_indices, dtype=torch.long)]
    stage1_query_emb = embeddings_for_faiss[stage1_query_modality][torch.as_tensor(query_stage1_indices, dtype=torch.long)]
    stage1_target_emb = embeddings_for_faiss[stage1_target_modality]

    # Compute full-database ranks for direct stage-1 and fused stage-2 comparison
    stage1_query_np = stage1_query_emb.detach().cpu().float().numpy()
    stage2_query_np = stage2_query_emb.detach().cpu().float().numpy()
    stage1_target_np = stage1_target_emb.detach().cpu().float().numpy()

    stage1_query_norm = np.linalg.norm(stage1_query_np, axis=1, keepdims=True) + 1e-12
    stage2_query_norm = np.linalg.norm(stage2_query_np, axis=1, keepdims=True) + 1e-12
    stage1_target_norm = np.linalg.norm(stage1_target_np, axis=1, keepdims=True) + 1e-12

    stage1_query_normalized = stage1_query_np / stage1_query_norm
    stage2_query_normalized = stage2_query_np / stage2_query_norm
    stage1_target_normalized = stage1_target_np / stage1_target_norm

    stage1_full_similarity = stage1_query_normalized @ stage1_target_normalized.T
    stage2_full_similarity = stage2_query_normalized @ stage1_target_normalized.T
    fused_full_similarity = fusion_alpha * stage1_full_similarity + (1.0 - fusion_alpha) * stage2_full_similarity

    stage1_correct_scores = stage1_full_similarity[np.arange(len(valid_query_ids)), gt_target_positions]
    stage1_ranks = 1 + np.sum(stage1_full_similarity > stage1_correct_scores[:, np.newaxis], axis=1)

    fused_correct_scores = fused_full_similarity[np.arange(len(valid_query_ids)), gt_target_positions]
    fused_ranks = 1 + np.sum(fused_full_similarity > fused_correct_scores[:, np.newaxis], axis=1)

    chain_rank_results["chain_stage1_full_rank"] = pd.DataFrame(
        {
            "material_id": valid_query_ids,
            "rank": stage1_ranks,
        }
    )
    chain_rank_results["chain_stage2_fused_full_rank"] = pd.DataFrame(
        {
            "material_id": valid_query_ids,
            "rank": fused_ranks,
        }
    )

    first_stage_scores, first_stage_indices = query(
        stage1_target_np,
        stage1_query_emb.detach().cpu().float().numpy(),
        [candidate_k],
        use_gpu=use_gpu,
        return_distances=True,
    )

    # Stage-1 metrics
    stage1_retrieved_ids = np.full((len(valid_query_ids), max_k), "", dtype=object)
    for i in range(len(valid_query_ids)):
        top_ids = [stage1_target_ids[idx] for idx in first_stage_indices[i, :max_k]]
        if top_ids:
            stage1_retrieved_ids[i, : min(len(top_ids), max_k)] = top_ids[:max_k]
    stage1_metrics = compute_retrieval_metrics(valid_query_ids, stage1_retrieved_ids, top_k)

    # Stage-2 hard rerank and fused rerank
    chain_hard_retrieved_ids = np.full((len(valid_query_ids), max_k), "", dtype=object)
    chain_fused_retrieved_ids = np.full((len(valid_query_ids), max_k), "", dtype=object)

    for i, _qid in enumerate(valid_query_ids):
        candidate_positions = [int(idx) for idx in first_stage_indices[i]]
        if not candidate_positions:
            continue

        candidate_emb = embeddings_for_faiss[stage1_target_modality][torch.as_tensor(candidate_positions, dtype=torch.long)]
        candidate_ids = [stage1_target_ids[idx] for idx in candidate_positions]
        stage1_sim = first_stage_scores[i]

        stage2_vec = stage2_query_emb[i].detach().cpu().float().numpy()
        cand_mat = candidate_emb.detach().cpu().float().numpy()
        stage2_norm = np.linalg.norm(stage2_vec) + 1e-12
        cand_norm = np.linalg.norm(cand_mat, axis=1) + 1e-12
        stage2_sim = (cand_mat @ stage2_vec) / (cand_norm * stage2_norm)

        hard_order = np.argsort(-stage2_sim, stable=True)
        hard_ranked_ids = [candidate_ids[j] for j in hard_order[:max_k]]
        if hard_ranked_ids:
            chain_hard_retrieved_ids[i, : min(len(hard_ranked_ids), max_k)] = hard_ranked_ids[:max_k]

        fused_scores = fusion_alpha * stage1_sim + (1.0 - fusion_alpha) * stage2_sim
        fused_order = np.argsort(-fused_scores, stable=True)
        fused_ranked_ids = [candidate_ids[j] for j in fused_order[:max_k]]
        if fused_ranked_ids:
            chain_fused_retrieved_ids[i, : min(len(fused_ranked_ids), max_k)] = fused_ranked_ids[:max_k]

    hard_metrics = compute_retrieval_metrics(valid_query_ids, chain_hard_retrieved_ids, top_k)
    fused_metrics = compute_retrieval_metrics(valid_query_ids, chain_fused_retrieved_ids, top_k)

    chain_metrics["chain_stage1"] = stage1_metrics
    chain_metrics["chain_stage2_hard_rerank"] = hard_metrics
    chain_metrics["chain_stage2_fused_rerank"] = fused_metrics

    LOGGER.info(f"Chain stage-1 metrics: {stage1_metrics}")
    LOGGER.info(f"Chain stage-2 hard rerank metrics: {hard_metrics}")
    LOGGER.info(f"Chain stage-2 fused rerank metrics (alpha={fusion_alpha:.2f}): {fused_metrics}")

    stage1_rank_df = chain_rank_results["chain_stage1_full_rank"]
    fused_rank_df = chain_rank_results["chain_stage2_fused_full_rank"]
    LOGGER.info(
        "Chain rank statistics - stage1: "
        f"mean={stage1_rank_df['rank'].mean():.2f}, median={stage1_rank_df['rank'].median():.2f}, "
        f"rank@1={(stage1_rank_df['rank'] == 1).sum() / len(stage1_rank_df) * 100:.2f}%"
    )
    LOGGER.info(
        "Chain rank statistics - fused stage2: "
        f"mean={fused_rank_df['rank'].mean():.2f}, median={fused_rank_df['rank'].median():.2f}, "
        f"rank@1={(fused_rank_df['rank'] == 1).sum() / len(fused_rank_df) * 100:.2f}%"
    )

    return chain_metrics, chain_rank_results


def compute_property_retrieval_metrics(
    aligned_df: pd.DataFrame,
    retrieved_result: dict[str, dict[str, np.ndarray]],
    top_k: list[int],
    property_name: str,
    property_data_path: str,
    retrieval_pair: str,
) -> dict[str, float]:
    """Compute property-aware retrieval recall for a given retrieval pair."""
    property_df = pd.read_csv(property_data_path)
    data_w_property = aligned_df.merge(property_df, on="material_id", how="inner")
    property_retrieval_metrics: dict[str, float] = {}

    if retrieval_pair not in retrieved_result:
        LOGGER.warning(f"Property retrieval skipped: retrieval pair '{retrieval_pair}' not found")
        return property_retrieval_metrics

    query_ids = retrieved_result[retrieval_pair]["material_ids"]
    retrieved_ids = retrieved_result[retrieval_pair]["retrieved_ids"]

    for k in top_k:
        correct = 0
        for i, query_id in enumerate(query_ids):
            property_value = data_w_property.loc[data_w_property["material_id"] == query_id][property_name]
            for retrieved_id in retrieved_ids[i, :k]:
                retrieved_property_value = data_w_property.loc[data_w_property["material_id"] == retrieved_id][property_name]
                if retrieved_property_value.equals(property_value):
                    correct += 1
                    break
        property_retrieval_metrics[f"Recall@{k}"] = correct / len(query_ids)

    return property_retrieval_metrics


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
    # Use appropriate dataloader
    dataloader_class = GeometricDataLoader if isinstance(dataset, GeometricDataset) else DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {modality}", leave=False):
            # Move batch to device - handle different data types
            if isinstance(batch, dict):
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch_data = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif hasattr(batch, "to"):
                batch_data = batch.to(device)
            else:
                batch_data = batch

            # Model is already in float32, no need for autocast
            embeddings = model.model.encode(modality, batch_data, with_projection=True)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def prepare_data(
    datamodule: MatBindDataModule,
    sub_val: bool = False,
    use_training_data_subset: bool = False,
    all_data: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    if datamodule.train_data is None or datamodule.val_data is None:
        raise ValueError("Datamodule train_data/val_data are not initialized. Call datamodule.setup() first.")

    if all_data:
        data = pd.concat([datamodule.train_data, datamodule.val_data], ignore_index=True)
        return data.reset_index(drop=True)
    elif sub_val:
        data = datamodule.val_data.sample(n=int(0.1 * len(datamodule.val_data)), random_state=seed)
        return data.reset_index(drop=True)
    data = (
        datamodule.train_data.sample(n=len(datamodule.val_data), random_state=datamodule.seed)
        if use_training_data_subset
        else datamodule.val_data
    )
    return data.reset_index(drop=True)


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
        datamodule: Data module with val_data
        modalities: list of modality names
        batch_size: Batch size for embedding
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
            # Create empty embeddings - dimension will be determined by first non-empty modality
            embeddings_dict[modality] = ModalityEmbeddings(
                embeddings=torch.empty(0),  # Will be reshaped if needed
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


def prepare_embeddings_for_retrieval(
    embeddings_dict: dict[str, ModalityEmbeddings],
    central_modality: str | None = None,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """
    Prepare data for FAISS retrieval by organizing embeddings and creating aligned dataframe.

    Args:
        embeddings_dict: dictionary of ModalityEmbeddings
        val_data: Original validation dataframe

    Returns:
        - dictionary mapping modality to embedding tensor
        - Aligned dataframe for retrieval (one row per material_id that has all modalities)
    """
    # Convert to simple dict for FAISS
    embeddings_for_faiss = {modality: emb_data.embeddings for modality, emb_data in embeddings_dict.items()}

    # Create aligned dataframe
    # For each modality, we have embeddings in the order of material_ids
    # We'll create a dataframe that has one row per embedding position
    aligned_data_parts = []

    for modality, emb_data in embeddings_dict.items():
        if len(emb_data) == 0:
            continue

        # Create a dataframe for this modality's embeddings
        modality_df = pd.DataFrame(
            {
                "material_id": emb_data.material_ids,
                modality: [True] * len(emb_data.material_ids),  # Marker that this modality exists
            }
        )

        aligned_data_parts.append(modality_df)

    # Merge to get final structure - each modality contributes its samples
    # We'll use the first modality as base
    if not aligned_data_parts:
        raise ValueError("No embeddings generated for any modality")

    # Start with all unique material IDs that appear in any modality
    # Preserve order by using dict (maintains insertion order in Python 3.7+)
    all_material_ids = {}
    for emb_data in embeddings_dict.values():
        for mid in emb_data.material_ids:
            all_material_ids[mid] = True

    aligned_df = pd.DataFrame({"material_id": list(all_material_ids.keys())})

    # For each modality, mark which material_ids have data
    for modality, emb_data in embeddings_dict.items():
        aligned_df[modality] = aligned_df["material_id"].isin(emb_data.material_ids)
        # Replace True with a placeholder value (FAISS script checks .notna())
        aligned_df.loc[aligned_df[modality], modality] = 1.0

    # Replace False with NaN for compatibility with existing retrieval code
    for modality in embeddings_dict:
        aligned_df.loc[~aligned_df[modality].astype(bool), modality] = pd.NA

    # If a central modality is specified, restrict the aligned dataframe to
    # rows that include the central modality. This mirrors the behavior of
    # dropping rows without the central modality at datamodule setup and
    # makes retrieval deterministic whether or not upstream dropna was used.
    if central_modality is not None:
        if central_modality in aligned_df.columns:
            before = aligned_df.shape[0]
            aligned_df = aligned_df.loc[aligned_df[central_modality].notna()].reset_index(drop=True)
            after = aligned_df.shape[0]
            LOGGER.info(f"Filtered aligned_df by central_modality='{central_modality}': {before} -> {after} rows")
        else:
            LOGGER.warning(f"central_modality '{central_modality}' not present in aligned dataframe columns")

    LOGGER.info(f"\nAligned dataframe shape: {aligned_df.shape}")
    LOGGER.info(f"Unique materials: {len(aligned_df)}")

    # Now build embeddings_for_faiss, reordering each modality's embeddings to
    # follow the order of material_ids in aligned_df for that modality. This
    # makes positional indexing in `faiss_retrieval` safe.

    embeddings_for_faiss: dict[str, torch.Tensor] = {}
    for modality, emb_data in embeddings_dict.items():
        orig_ids = [str(m).strip() for m in emb_data.material_ids]
        orig_emb = emb_data.embeddings

        # material ids for this modality as they appear in aligned_df
        if modality in aligned_df.columns:
            aligned_mod_ids = aligned_df.loc[aligned_df[modality].notna(), "material_id"].astype(str).str.strip().to_list()
        else:
            aligned_mod_ids = orig_ids

        id_to_pos = {mid: i for i, mid in enumerate(orig_ids)}
        positions = []
        missing = []
        for mid in aligned_mod_ids:
            if mid in id_to_pos:
                positions.append(id_to_pos[mid])
            else:
                missing.append(mid)

        if missing:
            LOGGER.warning(
                f"Some material_ids for modality '{modality}' present in aligned_df are missing from embeddings: sample {missing[:5]}"
            )

        if orig_emb.numel() == 0:
            embeddings_for_faiss[modality] = orig_emb
        else:
            if len(positions) == 0:
                try:
                    dim = orig_emb.shape[1]
                    embeddings_for_faiss[modality] = torch.empty((0, dim))
                except Exception:
                    embeddings_for_faiss[modality] = torch.empty(0)
            else:
                embeddings_for_faiss[modality] = orig_emb[positions]

    return embeddings_for_faiss, aligned_df


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
    """Always use float32 for inference to avoid dtype issues.
    Use .to() with explicit dtype to ensure conversion happens correctly.
    Fix encoder dtype attributes (some encoders store self.dtype).
    This is necessary because model creation with mixed_precision=True sets self.dtype=bfloat16.
    even though the loaded parameters are float32."""

    LOGGER.info(f"Converting model to float32 and moving to device {device} for inference")

    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    first_param = next(model.parameters())
    model_dtype = first_param.dtype
    LOGGER.info(f"Checkpoint model dtype: {model_dtype}")
    LOGGER.info("Scanning for modules with dtype attributes that need updating...")
    dtype_modules_found = 0
    for name, module in model.named_modules():
        if hasattr(module, "dtype") and module.dtype is not None:
            LOGGER.info(f"  Found module with dtype: {name}, dtype={module.dtype}")
            if module.dtype == torch.bfloat16:
                LOGGER.info(f"  - Changing {name}.dtype from bfloat16 to float32")
                module.dtype = torch.float32
                dtype_modules_found += 1

    if dtype_modules_found > 0:
        LOGGER.info(f"Updated {dtype_modules_found} encoder dtype attributes")
    else:
        LOGGER.info("No encoder dtype attributes needed updating")

    non_float32_params = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            LOGGER.warning(f"  Parameter {name} still has dtype {param.dtype}, converting...")
            param.data = param.data.float()
            non_float32_params += 1

    if non_float32_params > 0:
        LOGGER.info(f"Converted {non_float32_params} non-float32 parameters")

    final_param = next(model.parameters())
    LOGGER.info(f"Final model dtype: {final_param.dtype}, device: {final_param.device}")
    LOGGER.info("Model ready for inference")

    return model


def save_embeddings(
    embeddings_dict: dict[str, ModalityEmbeddings],
    save_location: Path,
):
    with save_location.open("wb") as f:
        pkl.dump(
            {
                "embeddings": {mod: emb_data.embeddings for mod, emb_data in embeddings_dict.items()},
                "metadata": {
                    mod: {"material_ids": emb_data.material_ids, "indices": emb_data.indices}
                    for mod, emb_data in embeddings_dict.items()
                },
            },
            f,
            protocol=pkl.HIGHEST_PROTOCOL,
        )


def load_embeddings(
    embeddings_path: Path,
) -> dict[str, ModalityEmbeddings]:
    with embeddings_path.open("rb") as f:
        data = pkl.load(f)

    embeddings_dict = {}
    for modality, emb_tensor in data["embeddings"].items():
        metadata = data["metadata"][modality]
        embeddings_dict[modality] = ModalityEmbeddings(
            embeddings=emb_tensor,
            material_ids=metadata["material_ids"],
            indices=metadata["indices"],
        )

    return embeddings_dict


def main_embed(
    training_config: DictConfig,
    ckpt_path: Path,
    results_save_location: Path,
    retrieval_batch_size: int,
    top_k: list[int],
    property_prediction: str | None = None,
    chain_query: bool = False,
    only_rank: bool = False,
    load_embeddings_from_file: bool = False,
):
    LOGGER.info(f"Using batch size: {retrieval_batch_size}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")

    LOGGER.info(f"Loading model from checkpoint: {ckpt_path}")

    model = load_model_from_checkpoint(
        training_config=training_config,
        ckpt_path=ckpt_path,
        device=device,
    )
    LOGGER.info("Model weights loaded successfully")

    model = set_model_to_eval_float32(model, device)

    datamodule: MatBindDataModule = hydra.utils.instantiate(training_config.data)
    datamodule.setup()
    datasets = get_retrieval_datasets(datamodule, training_config)

    for prefix, data in datasets:
        LOGGER.info(f"\n{'#' * 60}\nProcessing dataset: {prefix}\n{'#' * 60}")
        save_location = results_save_location / prefix
        save_location.mkdir(parents=True, exist_ok=True)

        embeddings_path = save_location / "embeddings.pkl"
        if load_embeddings_from_file and embeddings_path.exists():
            LOGGER.info(f"Loading embeddings from {embeddings_path}")
            embeddings_dict = load_embeddings(embeddings_path)
        else:
            embeddings_dict = embed_all_modalities(
                model=model,
                data=data,
                modality_to_dataset=datamodule.dataset_builder.modality_to_dataset,
                modalities=training_config.data.modalities,
                batch_size=retrieval_batch_size,
                device=device,
            )
            save_embeddings(
                embeddings_dict=embeddings_dict,
                save_location=embeddings_path,
            )
            LOGGER.info(f"Saved embeddings to {embeddings_path}")

        embeddings_for_faiss, aligned_df = prepare_embeddings_for_retrieval(
            embeddings_dict=embeddings_dict, central_modality=training_config.data.get("central_modality", None)
        )

        if only_rank:
            rank_results = comput_rank(
                embeddings=embeddings_for_faiss,
                indices=aligned_df,
                all_modalities=training_config.data.modalities,
            )
            rank_results_path = save_location / "rank_results.pkl"
            with rank_results_path.open("wb") as f:
                pkl.dump(rank_results, f, protocol=pkl.HIGHEST_PROTOCOL)
            LOGGER.info(f"Saved rank results to {rank_results_path}")
            return

        use_gpu = training_config.get("use_gpu_for_retrieval", True)
        retrieval_metrics, retrieved_result = faiss_retrieval(
            embeddings=embeddings_for_faiss,
            indices=aligned_df,
            all_modalities=training_config.data.modalities,
            top_k=top_k,
            use_gpu=use_gpu,
        )

        if chain_query:
            chain_metrics, chain_rank_results = run_chain_query_retrieval(
                embeddings_for_faiss=embeddings_for_faiss,
                aligned_df=aligned_df,
                top_k=top_k,
                use_gpu=use_gpu,
                config=training_config,
            )
            retrieval_metrics.update(chain_metrics)
            if chain_rank_results:
                chain_rank_path = save_location / "chain_rank_results.pkl"
                with chain_rank_path.open("wb") as f:
                    pkl.dump(chain_rank_results, f, protocol=pkl.HIGHEST_PROTOCOL)
                LOGGER.info(f"Saved chain rank results to {chain_rank_path}")

                for rank_name, rank_df in chain_rank_results.items():
                    rank_csv_path = save_location / f"{rank_name}.csv"
                    rank_df.to_csv(rank_csv_path, index=False)
                    LOGGER.info(f"Saved chain rank csv to {rank_csv_path}")

        if property_prediction is not None:
            property_cfg = training_config.get("property_prediction", {})
            retrieval_pair = property_cfg.get("retrieval_pair", "text_crystal_structure")
            property_data_path = property_cfg.get("data_path", str(Path(training_config.data.data_dir) / "symmetry_data.csv"))
            property_metrics = compute_property_retrieval_metrics(
                aligned_df=aligned_df,
                retrieved_result=retrieved_result,
                top_k=top_k,
                property_name=property_prediction,
                property_data_path=property_data_path,
                retrieval_pair=retrieval_pair,
            )
            if property_metrics:
                retrieval_metrics[f"property_{property_prediction}"] = property_metrics
                LOGGER.info(f"Property-based retrieval metrics for '{property_prediction}': {property_metrics}")

        retrieval_metrics_df = pd.DataFrame(retrieval_metrics)
        retrieval_metrics_path = save_location / "retrieval_metrics.csv"
        retrieval_metrics_df.to_csv(retrieval_metrics_path)

        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info("RESULTS")
        LOGGER.info(f"{'=' * 60}")
        LOGGER.info(f"Saved retrieval metrics to {retrieval_metrics_path}")
        LOGGER.info(f"\nRetrieval metrics:\n{retrieval_metrics_df.to_string()}")


def get_checkpoint_path(ckpt_path: str | None, training_dir: Path) -> Path:
    if ckpt_path is not None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided checkpoint path {ckpt_path} does not exist.")
        LOGGER.info(f"Using provided checkpoint path: {path}")
        return path

    check_point_dir = training_dir / "checkpoints"
    checkpoints = list(check_point_dir.glob("*.ckpt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in provided checkpoint directory {check_point_dir}")

    non_last_checkpoints = list(filter(lambda p: p.stem != "last", checkpoints))
    if non_last_checkpoints:
        if num_last := len(non_last_checkpoints) > 1:
            newline = "\n  - "
            raise ValueError(
                f"Found {num_last} non-last checkpoints in {check_point_dir}, please specify one of the following ckpt_paths explicitly:{newline}{newline.join(str(p) for p in non_last_checkpoints)}"
            )

        LOGGER.info(f"Found checkpoint {non_last_checkpoints[0]}, using it.")
        return non_last_checkpoints[0]

    last_ckpt = list(filter(lambda p: p.stem == "last", checkpoints))

    if last_ckpt:
        LOGGER.warning(f"Only 'last.ckpt' found in {check_point_dir}, using it.")
        return last_ckpt[0]

    raise ValueError(f"Could not determine checkpoint to use from directory {check_point_dir}.")


@hydra.main(version_base="1.3", config_path="../configs", config_name="retrieval.yaml")
def main(config: RetrievalConfig):
    # training_dir = Path(config.training_dir)

    # if not training_dir.exists():
    #     raise FileNotFoundError(f"Training directory {training_dir} does not exist.")

    # training_config = OmegaConf.load(training_dir / ".hydra" / "config.yaml")
    training_config = config
    # training_config["data"]["dataset_builder"]["modality_to_dataset"]["text"]["partial_text"] = False
    if config.ckpt_path is None:
        raise ValueError("`ckpt_path` must be provided for retrieval.")
    ckpt_path = Path(config.ckpt_path)
    results_save_location = Path(config.results_save_location)
    results_save_location.mkdir(parents=True, exist_ok=True)

    # LOGGER.info("Using training configuration:")
    # print_config_tree(training_config)

    main_embed(
        training_config=training_config,
        ckpt_path=ckpt_path,
        results_save_location=results_save_location,
        retrieval_batch_size=config.retrieval_batch_size,
        top_k=config.top_k,
        chain_query=True,
        only_rank=False,
        load_embeddings_from_file=False,  # Set to True to skip embedding and load from file (make sure embeddings_path is correct in config)
    )


if __name__ == "__main__":
    main()
