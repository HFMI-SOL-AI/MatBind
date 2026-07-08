import itertools

import faiss
import numpy as np
import pandas as pd
import torch

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


def compute_retrieval_metrics(
    query_ids: list[str],
    retrieved_ids: np.ndarray,
    top_k: list[int],
) -> dict:
    """
    Compute Recall@K metrics.

    Args:
        query_ids: List of query material IDs (ground truth)
        retrieved_ids: Array of shape (n_queries, n_results) with retrieved material IDs
        top_k: List of K values to compute Recall@K for

    Returns:
        Dictionary with Recall@K metrics
    """
    retrieval_metrics = {}

    for k in top_k:
        correct = 0
        for i, query_id in enumerate(query_ids):
            if query_id in retrieved_ids[i, :k]:
                correct += 1
        retrieval_metrics[f"Recall@{k}"] = correct / len(query_ids)

    return retrieval_metrics


def faiss_retrieval(
    embeddings: dict[str, torch.Tensor],
    indices: pd.DataFrame,
    all_modalities: list[str],
    top_k: list[int],
    use_gpu: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    """
    Perform cross-modal retrieval using FAISS for efficient similarity search.

    This function expects:
    - embeddings: Dict mapping modality -> tensor of embeddings (one per non-null sample)
    - indices: DataFrame where each row corresponds to a material_id, with columns for each modality
              (values are 1.0 if available, NaN if not)

    The key assumption: embeddings[modality][i] corresponds to the i-th non-null sample
    for that modality in the indices dataframe (in order of appearance).

    Args:
        embeddings: Dictionary mapping modality names to embedding tensors
        indices: DataFrame with material_id and modality availability columns
        all_modalities: List of all modality names
        top_k: List of K values for Recall@K metrics
        use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)

    Returns:
        Dictionary of retrieval metrics for each modality pair
    """

    retrieval_metrics = {}
    retrieval_results = {}

    LOGGER.info("Starting FAISS-based retrieval")

    for modality_1 in all_modalities:
        for modality_2 in all_modalities:
            if modality_2 == modality_1:
                continue

            if embeddings[modality_1].size(0) == 0 or embeddings[modality_2].size(0) == 0:
                LOGGER.warning(f"Skipping {modality_1} -> {modality_2}: empty embeddings")
                continue

            LOGGER.info(f"\n{modality_1} -> {modality_2} retrieval")

            # Find material_ids that have both modalities
            # THIS IS IMPORTANT -> sinc enot all modalities have the same length
            both_available = indices[[modality_1, modality_2, "material_id"]].dropna(subset=[modality_1, modality_2])

            n_pairs = len(both_available)
            if n_pairs == 0:
                LOGGER.warning(f"No samples with both {modality_1} and {modality_2}")
                continue

            LOGGER.info(f"Found {n_pairs} samples with both modalities")

            material_ids = both_available["material_id"].to_list()

            mod1_material_ids = indices[indices[modality_1].notna()]["material_id"].to_list()
            mod2_material_ids = indices[indices[modality_2].notna()]["material_id"].to_list()

            mod1_id_to_pos = {mid: i for i, mid in enumerate(mod1_material_ids)}
            mod2_id_to_pos = {mid: i for i, mid in enumerate(mod2_material_ids)}

            mod1_positions = [mod1_id_to_pos[mid] for mid in material_ids]
            mod2_positions = [mod2_id_to_pos[mid] for mid in material_ids]

            database_embeds = embeddings[modality_1][mod1_positions].detach().cpu().float().numpy()
            query_embeds = embeddings[modality_2][mod2_positions].detach().cpu().float().numpy()

            LOGGER.info(f"Database embeddings: {database_embeds.shape}")
            LOGGER.info(f"Query embeddings: {query_embeds.shape}")

            indices_faiss = query(database_embeds, query_embeds, top_k, use_gpu)

            # Map indices back to material IDs
            # indices_faiss contains positions in database_embeds
            # database_embeds[i] corresponds to material_ids[i]
            retrieved_ids = np.array([[material_ids[idx] for idx in row] for row in indices_faiss])

            metrics = compute_retrieval_metrics(material_ids, retrieved_ids, top_k)
            retrieval_metrics[f"{modality_1}_{modality_2}"] = metrics
            retrieval_results[f"{modality_1}_{modality_2}"] = {
                "retrieved_ids": retrieved_ids,
                "material_ids": material_ids,
            }

            LOGGER.info(f"Metrics: {metrics}")

    LOGGER.info("Retrieval complete")

    return retrieval_metrics, retrieval_results


def query(database_embeds, query_embeds, top_k, use_gpu=False, return_distances: bool = False):
    faiss.normalize_L2(database_embeds)
    faiss.normalize_L2(query_embeds)

    d = database_embeds.shape[1]  # embedding dimension
    index = faiss.IndexFlatIP(d)  # Inner Product (cosine similarity after normalization)

    if use_gpu and faiss.get_num_gpus() > 0:
        LOGGER.info("Using GPU for FAISS")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(database_embeds)

    k = max(top_k)
    distances, indices_faiss = index.search(query_embeds, k)
    if return_distances:
        return distances, indices_faiss
    return indices_faiss


def comput_rank(
    embeddings: dict[str, torch.Tensor],
    indices: pd.DataFrame,
    all_modalities: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compute rank of correct results for each query-database pair.

    Args:
        embeddings: Dictionary mapping modality names to embedding tensors
        indices: DataFrame with material_id and modality availability columns
        all_modalities: List of all modality names

    Returns:
        Dictionary mapping modality pair names to DataFrames containing rank information
    """
    rank_results = {}

    for modality_1, modality_2 in itertools.product(all_modalities, all_modalities):
        LOGGER.info(f"\n{modality_1} -> {modality_2} rank computation")
        if modality_2 == modality_1:
            LOGGER.info("Skipping same modality pair")
            continue

        if embeddings[modality_1].size(0) == 0 or embeddings[modality_2].size(0) == 0:
            LOGGER.warning(f"Skipping {modality_1} -> {modality_2}: empty embeddings")
            continue

        # Find material_ids that have both modalities
        both_available = indices[[modality_1, modality_2, "material_id"]].dropna(subset=[modality_1, modality_2])

        n_pairs = len(both_available)
        if n_pairs == 0:
            LOGGER.warning(f"No samples with both {modality_1} and {modality_2}")
            continue

        LOGGER.info(f"Found {n_pairs} samples with both modalities")

        material_ids = both_available["material_id"].to_list()

        mod1_material_ids = indices[indices[modality_1].notna()]["material_id"].to_list()
        mod2_material_ids = indices[indices[modality_2].notna()]["material_id"].to_list()

        mod1_id_to_pos = {mid: i for i, mid in enumerate(mod1_material_ids)}
        mod2_id_to_pos = {mid: i for i, mid in enumerate(mod2_material_ids)}

        mod1_positions = [mod1_id_to_pos[mid] for mid in material_ids]
        mod2_positions = [mod2_id_to_pos[mid] for mid in material_ids]

        database_embeds = embeddings[modality_1][mod1_positions].detach().cpu().float().numpy()
        query_embeds = embeddings[modality_2][mod2_positions].detach().cpu().float().numpy()

        LOGGER.info(f"Database embeddings: {database_embeds.shape}")
        LOGGER.info(f"Query embeddings: {query_embeds.shape}")

        # Normalize embeddings
        database_norm = np.linalg.norm(database_embeds, axis=1, keepdims=True) + 1e-12
        query_norm = np.linalg.norm(query_embeds, axis=1, keepdims=True) + 1e-12

        database_normalized = database_embeds / database_norm
        query_normalized = query_embeds / query_norm

        # Compute similarity matrix
        similarity = query_normalized @ database_normalized.T  # Shape: (n_queries, n_database)

        # For each query, find the rank of the correct match (vectorized)
        # Get similarity scores for correct matches (diagonal elements)
        correct_scores = np.diag(similarity)  # Shape: (n_queries,)

        # Count how many database samples have higher similarity than the correct match
        # Expand dims to allow broadcasting: (n_queries, 1) vs (n_queries, n_database)
        correct_scores_expanded = correct_scores[:, np.newaxis]
        ranks = 1 + np.sum(similarity > correct_scores_expanded, axis=1)  # +1 because rank starts at 1

        # Create DataFrame directly from ranks
        rank_df = pd.DataFrame({"rank": ranks})
        pair_name = f"{modality_1}_{modality_2}"
        rank_results[pair_name] = rank_df

        LOGGER.info(f"Rank statistics for {pair_name}:")
        LOGGER.info(f"  Mean rank: {rank_df['rank'].mean():.2f}")
        LOGGER.info(f"  Median rank: {rank_df['rank'].median():.2f}")
        LOGGER.info(f"  Min rank: {rank_df['rank'].min()}")
        LOGGER.info(f"  Max rank: {rank_df['rank'].max()}")
        LOGGER.info(f"  Rank@1: {(rank_df['rank'] == 1).sum() / len(rank_df) * 100:.2f}%")

    return rank_results
