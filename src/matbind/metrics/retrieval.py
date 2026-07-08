import chromadb
import contextlib
from collections import Counter
import numpy as np
import pandas as pd
import torch

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


def compute_retrieval_metrics_from_query(
    ids: list[str],
    retrieved_ids: list[str],
    top_k: list[int],
) -> dict:
    retrieval_metrics_entire_db = {k: np.zeros(len(ids)) for k in top_k}
    for k in top_k:
        for i, id_ in enumerate(ids):
            if id_ in retrieved_ids["ids"][i][:k]:
                retrieval_metrics_entire_db[k][i] = 1
    return {f"Recall@{k}": np.mean(retrieval_metrics_entire_db[k]) for k in top_k}


def full_database_retrieval(
    embeddings: dict[str, torch.Tensor],
    indices: pd.DataFrame,
    other_modalities: list[str],
    central_modality: str,
    top_k: list[int],
    use_approx_index: bool = False,
) -> dict[str, dict[str, float]]:
    all_modalities = [central_modality, *other_modalities]
    # embeddings for each modality are expected to be a tuple:
    # (tensor_of_embeddings, list_of_material_ids_corresponding_to_rows_in_tensor)
    # e.g. embeddings[modality] -> (torch.Tensor[num_rows, dim], List[str][num_rows])
    if isinstance(embeddings.get(central_modality), tuple):
        LOGGER.info(
            f"embedding have length {embeddings[central_modality][0].shape} (unpacked from tuple)"
        )
    else:
        LOGGER.info(f"embedding have length {embeddings[central_modality].shape}")
    # Canonicalize material ids in the DataFrame to ensure consistent matching
    if "material_id" in indices.columns:
        indices["material_id"] = indices["material_id"].astype(str).str.strip()
    LOGGER.info(f"dataset have length {len(indices)}")

    retrieval_metrics = {}
    client = chromadb.Client() if use_approx_index else None
    store_retrieved_ids_for_modality_pair = {}

    for modality_1 in all_modalities:
        modalities_retrieval = {}
        for modality_2 in all_modalities:
            if modality_2 != modality_1:
                collection = None
                if use_approx_index:
                    with contextlib.suppress(Exception):
                        client.delete_collection(f"{modality_1}_embeds")
                    collection = client.create_collection(
                        name=f"{modality_1}_embeds",
                        metadata={
                            "hnsw:space": "cosine",
                            "hnsw:M": 32,
                            "hnsw:construction_ef": 1024,
                            "hnsw:search_ef": 2000,
                        },
                    )

                # Determine rows where both modalities exist (we will use the same set
                # of rows for the DB ( modality_1 ) and for queries ( modality_2 ) so
                # ids and embeddings remain aligned).
                mask_both = indices[[modality_1, modality_2]].dropna().index.to_list()

                # Build per-modality positional maps.
                # Each modality's embeddings are provided as a tuple
                # (emb_tensor, mids_list) where mids_list contains the
                # material ids aligned with rows of emb_tensor. We map
                # material_id -> position in that modality's embedding
                # tensor. Then we restrict to rows where both modalities
                # have embeddings (by material id) to keep alignment.
                rows_both = mask_both  # original DataFrame indices where both exist

                # Unpack embeddings -> (tensor, mids) if tuple, otherwise
                # fall back to old behavior assuming tensors aligned to
                # DataFrame order (less common now).
                mod1_emb_tuple = embeddings.get(modality_1)
                mod2_emb_tuple = embeddings.get(modality_2)

                # prepare defaults so static analyzers know variables exist
                mod1_mids = []
                mod2_mids = []

                if isinstance(mod1_emb_tuple, tuple):
                    # mod1_emb_tuple -> (tensor, mids); extract and normalize mids
                    mod1_mids_raw = mod1_emb_tuple[1]
                    mod1_mids = [str(m).strip() for m in mod1_mids_raw]
                    pos_map_mod1 = {mid: pos for pos, mid in enumerate(mod1_mids)}
                else:
                    # legacy: positional map from DataFrame index
                    mod1_rows = indices[modality_1].dropna().index.to_list()
                    pos_map_mod1 = {orig_idx: pos for pos, orig_idx in enumerate(mod1_rows)}

                if isinstance(mod2_emb_tuple, tuple):
                    mod2_mids_raw = mod2_emb_tuple[1]
                    mod2_mids = [str(m).strip() for m in mod2_mids_raw]
                    pos_map_mod2 = {mid: pos for pos, mid in enumerate(mod2_mids)}
                else:
                    mod2_rows = indices[modality_2].dropna().index.to_list()
                    pos_map_mod2 = {orig_idx: pos for pos, orig_idx in enumerate(mod2_rows)}

                # Material ids for rows_both (in same order as rows_both)
                mids_for_rows = indices["material_id"].loc[rows_both].to_list()
                # normalize mids from DataFrame
                mids_for_rows = [str(m).strip() for m in mids_for_rows]

                # Diagnostic logging: compare embedding mids vs DataFrame mids
                try:
                    if isinstance(mod1_emb_tuple, tuple):
                        set_emb1 = set(mod1_mids)
                        set_df1 = set(indices.loc[indices[modality_1].notna(), "material_id"].astype(str).str.strip().to_list())
                        inter1 = len(set_emb1 & set_df1)
                        LOGGER.info(
                            f"Modality {modality_1}: emb_count={len(mod1_mids)}, df_count={len(set_df1)}, intersection={inter1}"
                        )
                    if isinstance(mod2_emb_tuple, tuple):
                        set_emb2 = set(mod2_mids)
                        set_df2 = set(indices.loc[indices[modality_2].notna(), "material_id"].astype(str).str.strip().to_list())
                        inter2 = len(set_emb2 & set_df2)
                        LOGGER.info(
                            f"Modality {modality_2}: emb_count={len(mod2_mids)}, df_count={len(set_df2)}, intersection={inter2}"
                        )
                except Exception:
                    # best-effort diagnostics; don't fail retrieval for logging issues
                    LOGGER.debug("Failed to compute diagnostic set intersections for modalities")

                # Filter to only rows where both modalities actually have an
                # embedding (present in their mids lists). Keep lists aligned.
                filtered_rows = []
                pos_indices_mod1 = []
                pos_indices_mod2 = []
                filtered_mids = []
                for orig_idx, mid in zip(rows_both, mids_for_rows, strict=True):
                    try:
                        p1 = pos_map_mod1[mid]
                        p2 = pos_map_mod2[mid]
                    except Exception:
                        # Skip this row if either modality lacks the mid mapping
                        continue
                    filtered_rows.append(orig_idx)
                    pos_indices_mod1.append(p1)
                    pos_indices_mod2.append(p2)
                    filtered_mids.append(mid)

                    # Diagnostics: check for duplicate mids in embedding lists
                    if isinstance(mod1_emb_tuple, tuple):
                        dup1 = [m for m, cnt in Counter(mod1_mids).items() if cnt > 1]
                        if dup1:
                            LOGGER.warning(
                                f"Duplicate mids found in embeddings for {modality_1}: sample {dup1[:5]}"
                            )
                    if isinstance(mod2_emb_tuple, tuple):
                        dup2 = [m for m, cnt in Counter(mod2_mids).items() if cnt > 1]
                        if dup2:
                            LOGGER.warning(
                                f"Duplicate mids found in embeddings for {modality_2}: sample {dup2[:5]}"
                            )

                    # Diagnostics: verify that the selected embedding positions map back
                    # to the same material ids we expect (ids list). If this fails,
                    # retrieval will silently be misaligned.
                    try:
                        if isinstance(mod1_emb_tuple, tuple):
                            mapped_mids1 = [mod1_mids[p] for p in pos_indices_mod1]
                            mismatches1 = [i for i, (a, b) in enumerate(zip(mapped_mids1, filtered_mids, strict=True)) if a != b]
                            if mismatches1:
                                sample = mismatches1[:5]
                                LOGGER.error(
                                    f"Alignment mismatch for {modality_1} at positions (sample indices): {sample} -> "
                                    f"(emb_mid, expected_id): {[ (mapped_mids1[i], filtered_mids[i]) for i in sample ]}"
                                )
                        if isinstance(mod2_emb_tuple, tuple):
                            mapped_mids2 = [mod2_mids[p] for p in pos_indices_mod2]
                            mismatches2 = [i for i, (a, b) in enumerate(zip(mapped_mids2, filtered_mids, strict=True)) if a != b]
                            if mismatches2:
                                sample = mismatches2[:5]
                                LOGGER.error(
                                    f"Alignment mismatch for {modality_2} at positions (sample indices): {sample} -> "
                                    f"(emb_mid, expected_id): {[ (mapped_mids2[i], filtered_mids[i]) for i in sample ]}"
                                )
                    except Exception:
                        LOGGER.debug("Failed to run alignment diagnostics (non-fatal)")

                if len(filtered_rows) != len(rows_both):
                    LOGGER.warning(
                        f"Skipped {len(rows_both) - len(filtered_rows)} rows because a modality lacked an embedding mid mapping"
                    )

                # central rows and material ids (in the same order as filtered_rows)
                central_modality_data = indices[central_modality].loc[filtered_rows].to_list()
                ids = filtered_mids

                LOGGER.info(f"{modality_1} {modality_2} retrieval")
                LOGGER.info(f"{len(central_modality_data)}")

                batch_size = 5000  # Use a size smaller than the max limit
                def chunk_data(data, chunk_size):
                    """Split data into smaller chunks"""
                    for i in range(0, len(data), chunk_size):
                        yield data[i:i + chunk_size]

                # Prepare DB and query arrays (NumPy) selecting positions matching filtered rows.
                if isinstance(mod1_emb_tuple, tuple):
                    mod1_array = mod1_emb_tuple[0].detach().cpu().float().numpy()[pos_indices_mod1]
                else:
                    mod1_array = embeddings[modality_1].detach().cpu().float().numpy()[pos_indices_mod1]

                if isinstance(mod2_emb_tuple, tuple):
                    query_array = mod2_emb_tuple[0].detach().cpu().float().numpy()[pos_indices_mod2]
                else:
                    query_array = embeddings[modality_2].detach().cpu().float().numpy()[pos_indices_mod2]

                if use_approx_index:
                    # Add DB embeddings to chroma in batches (to avoid large memory ops)
                    for i, batch in enumerate(chunk_data(mod1_array, batch_size)):
                        batch_ids = ids[i * batch_size : (i + 1) * batch_size]
                        collection.add(
                            embeddings=batch,
                            ids=batch_ids,
                        )

                    # Query chroma collection (approximate, HNSW)
                    modalities_retrieval[modality_2] = collection.query(
                        query_embeddings=query_array,
                        n_results=max(top_k),
                        include=["distances"],
                    )
                else:
                    # Exact, deterministic retrieval by cosine similarity (brute force)
                    # Normalize vectors and compute dot-product (cosine similarity)
                    def l2_normalize(a: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
                        norms = np.linalg.norm(a, axis=axis, keepdims=True)
                        return a / np.maximum(norms, eps)

                    db_norm = l2_normalize(mod1_array)
                    q_norm = l2_normalize(query_array)
                    # cosine similarities = q_norm @ db_norm.T
                    sims = np.matmul(q_norm, db_norm.T)
                    # for each query, get indices sorted by descending similarity
                    topk = np.argsort(-sims, axis=1)[:, : max(top_k)]
                    retrieved_ids = [[ids[idx] for idx in row] for row in topk]
                    retrieved_dists = [[float(sims[i, idx]) for idx in row] for i, row in enumerate(topk)]
                    modalities_retrieval[modality_2] = {"ids": retrieved_ids, "distances": retrieved_dists}
                retrieval_metrics[f"{modality_1}_{modality_2}"] = compute_retrieval_metrics_from_query(
                    ids,
                    modalities_retrieval[modality_2],
                    top_k=top_k,
                )
                store_retrieved_ids_for_modality_pair[f"{modality_1}-{modality_2}"] = ids
                LOGGER.info(
                    f"Retrieval metrics for {modality_1} and {modality_2}:\
                    {retrieval_metrics[f'{modality_1}_{modality_2}']}"
                )
    return retrieval_metrics
