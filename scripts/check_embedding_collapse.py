#!/usr/bin/env python3
"""Check embedding collapse diagnostics.

This script can either analyze a saved embeddings pickle file or generate
embeddings using the project's embedding pipeline (Hydra) and then run
diagnostics on a chosen modality.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Any

import hydra
import numpy as np
from dotenv import load_dotenv
import rootutils

load_dotenv()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_array(obj: Any, key: str | None = None) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        if key is not None:
            if key in obj:
                return np.asarray(obj[key])
            raise KeyError(f"Key '{key}' not found in pickle dict. Available keys: {list(obj.keys())}")
        for candidate in ("embeddings", "emb", "vectors", "X"):
            if candidate in obj:
                return np.asarray(obj[candidate])
        values = list(obj.values())
        if values and isinstance(values[0], (list, tuple, np.ndarray)):
            return np.asarray(values)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)
    raise ValueError("Could not extract embeddings array from the provided pickle file.")


def safe_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def sample_pairwise_cosine(emb: np.ndarray, n_samples: int = 20000, seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    n = emb.shape[0]
    if n < 2:
        return np.array([])
    unit = safe_normalize(emb)
    max_pairs = n * (n - 1) // 2
    n_samples = int(min(n_samples, max_pairs))
    sims = []
    for _ in range(n_samples):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        sims.append(float(np.dot(unit[i], unit[j])))
    return np.array(sims)


def explained_variance_svd(emb: np.ndarray, n_top: int = 10):
    X = emb.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        sample = X if X.shape[0] <= 2000 else X[np.random.choice(X.shape[0], 2000, replace=False)]
        _, s, _ = np.linalg.svd(sample, full_matrices=False)
    eig = s * s
    explained = eig / float(eig.sum())
    ratio_top = float(explained[0]) if explained.size > 0 else 0.0
    cumsum = np.cumsum(explained)
    n90 = int(np.searchsorted(cumsum, 0.90) + 1) if cumsum.size > 0 else 0
    return explained[:n_top], ratio_top, n90


def nearest_neighbor_distances(emb: np.ndarray, sample_n: int = 1000) -> np.ndarray:
    n = emb.shape[0]
    if n < 2:
        return np.array([])
    idx = np.arange(n)
    if n > sample_n:
        idx = np.random.choice(n, size=sample_n, replace=False)
    sample = emb[idx]
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(emb)
        distances, _ = nbrs.kneighbors(sample)
        return distances[:, 1]
    except Exception:
        dists = []
        for v in sample:
            dif = emb - v
            ds = np.sqrt((dif * dif).sum(axis=1))
            ds = np.sort(ds)
            if ds.size >= 2:
                dists.append(ds[1])
        return np.array(dists)



def summarize(emb: np.ndarray, pair_sims: np.ndarray, nn_dists: np.ndarray) -> None:
    n, d = emb.shape
    norms = np.linalg.norm(emb, axis=1)
    print(f"Shape: {n} x {d}")
    print(f"dtype: {emb.dtype}")
    print("-- norms --")
    print(f"min {float(norms.min()):.6f}, max {float(norms.max()):.6f}, mean {float(norms.mean()):.6f}, std {float(norms.std()):.6f}")
    print("-- pairwise cosine similarity (sampled) --")
    if pair_sims.size:
        print(f"count {pair_sims.size}, mean {pair_sims.mean():.6f}, std {pair_sims.std():.6f}")
        for p in (50, 90, 95, 99):
            print(f"{p}th percentile: {np.percentile(pair_sims, p):.6f}")
        for thresh in (0.9, 0.95, 0.99):
            frac = (pair_sims > thresh).mean()
            print(f"fraction > {thresh}: {frac:.4f}")
    else:
        print("not enough data to compute pairwise similarities")
    print("-- SVD / PCA explained variance --")
    explained_top, ratio_top, n90 = explained_variance_svd(emb, n_top=min(20, d))
    print(f"top component explained variance fraction: {ratio_top:.6f}")
    print(f"components needed for 90% explained variance: {n90}")
    print(f"first {explained_top.size} explained fractions: {', '.join(f'{x:.4f}' for x in explained_top)}")
    print("-- nearest neighbor distances (sampled) --")
    if nn_dists.size:
        print(f"count {nn_dists.size}, mean {nn_dists.mean():.6f}, std {nn_dists.std():.6f}")
        print(f"min {nn_dists.min():.6f}, 25% {np.percentile(nn_dists,25):.6f}, median {np.median(nn_dists):.6f}, 75% {np.percentile(nn_dists,75):.6f}, max {nn_dists.max():.6f}")
    else:
        print("not enough data to compute nearest neighbor distances")


def analyze_numpy_embeddings(emb: np.ndarray, pair_samples: int = 20000, nn_sample: int = 1000, plot_prefix: str | None = None) -> None:
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    emb = np.asarray(emb)
    if emb.size == 0:
        raise SystemExit("empty embeddings array")
    pair_sims = sample_pairwise_cosine(emb, n_samples=pair_samples)
    nn_dists = nearest_neighbor_distances(emb, sample_n=nn_sample)
    # compute covariance and effective rank
    X = emb.astype(np.float64)
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    n_samples_cov = Xc.shape[0]
    # covariance (D x D)
    cov = (Xc.T @ Xc) / float(max(1, n_samples_cov))
    # eigenvalues of covariance
    try:
        eigs = np.linalg.eigvalsh(cov)
    except Exception:
        eigs = np.linalg.svd(Xc, compute_uv=False) ** 2

    # numerical stability: clip negatives to zero
    eigs = np.clip(eigs, a_min=0.0, a_max=None)
    total = float(eigs.sum()) if eigs.sum() > 0 else 1.0
    p = eigs / total
    # effective rank = exp(entropy)
    eps = 1e-12
    entropy = -float(np.sum(np.where(p > 0, p * np.log(p + eps), 0.0)))
    effective_rank = float(np.exp(entropy)) if entropy >= 0 else 0.0

    summarize(emb, pair_sims, nn_dists)

    print("-- covariance / effective rank --")
    print(f"covariance shape: {cov.shape}, total variance: {total:.6f}")
    print(f"effective rank (exp entropy): {effective_rank:.4f}")
    # top eigenvalues
    topk = min(10, eigs.size)
    if topk > 0:
        top_eigs = eigs[::-1][:topk]
        print(f"top {topk} eigenvalues: {', '.join(f'{v:.6f}' for v in top_eigs)}")
    if plot_prefix:
        data = {"norms": np.linalg.norm(emb, axis=1)}
        if pair_sims.size:
            data["cosine"] = pair_sims
        if nn_dists.size:
            data["nn_dist"] = nn_dists
        maybe_plot(data, plot_prefix)
    # return cov and effective rank for callers that want to save or further inspect
    return {"cov": cov, "effective_rank": effective_rank, "eigenvalues": eigs}


def _unit_vectors(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def compute_modality_gap(embeddings: Dict[str, np.ndarray], sample_n: int = 1000, seed: int | None = 0) -> Dict[Tuple[str, str], dict]:
    """Compute modality-gap diagnostics between pairs of modalities.

    For each unordered pair (A, B) the diagnostics include:
    - paired_cosine: per-sample cosine similarity between aligned examples (if lengths match)
      and summary stats (mean, std, percentiles).
    - centroid_cosine: cosine similarity between mean embedding vectors of the two modalities.
    - mean_norm_diff: mean absolute difference of norms between modalities for paired examples (if aligned).
    - cross_nn: on a sampled subset, for each sample in A compute its nearest neighbor in B (by cosine)
      and report mean nearest-similarity and, when aligned and same length, the fraction of queries
      whose nearest neighbor is the true paired item (match_rate).

    Returns a dict keyed by (modA, modB) -> stats dict.
    """
    rng = random.Random(seed)
    keys = list(embeddings.keys())
    results: dict[tuple[str, str], dict] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            Ea = np.asarray(embeddings[a])
            Eb = np.asarray(embeddings[b])
            stats: dict = {}
            if Ea.size == 0 or Eb.size == 0:
                stats["note"] = "one or both embeddings empty; skipped"
                results[(a, b)] = stats
                continue

            # paired statistics when lengths match
            n_a = Ea.shape[0]
            n_b = Eb.shape[0]
            unit_a = _unit_vectors(Ea)
            unit_b = _unit_vectors(Eb)

            if n_a == n_b:
                paired = np.sum(unit_a * unit_b, axis=1)
                stats["paired_count"] = int(paired.size)
                stats["paired_mean"] = float(np.mean(paired))
                stats["paired_std"] = float(np.std(paired))
                for p in (50, 90, 95, 99):
                    stats[f"paired_p{p}"] = float(np.percentile(paired, p))
                stats["paired_frac_gt_0.9"] = float((paired > 0.9).mean())
                # norm differences
                norms_a = np.linalg.norm(Ea, axis=1)
                norms_b = np.linalg.norm(Eb, axis=1)
                stats["mean_abs_norm_diff"] = float(np.mean(np.abs(norms_a - norms_b)))
            else:
                stats["paired_note"] = f"length mismatch (A={n_a}, B={n_b}); paired stats skipped"

            # centroid similarity
            ma = unit_a.mean(axis=0)
            mb = unit_b.mean(axis=0)
            ma = ma / float(max(1e-12, np.linalg.norm(ma)))
            mb = mb / float(max(1e-12, np.linalg.norm(mb)))
            stats["centroid_cosine"] = float(np.dot(ma, mb))

            # cross-modal nearest neighbor stats (sampled)
            sample_size = min(int(sample_n), n_a)
            if sample_size <= 0:
                stats["cross_nn_note"] = "no samples for cross NN"
                results[(a, b)] = stats
                continue

            idx = np.asarray(rng.sample(range(n_a), sample_size)) if sample_size < n_a else np.arange(n_a)

            # compute similarities from A_samples -> all B
            # shape (sample_size, n_b)
            sims = unit_a[idx] @ unit_b.T
            # nearest neighbor similarity and argmax
            nn_sim = sims.max(axis=1)
            nn_idx = sims.argmax(axis=1)
            stats["cross_nn_mean_sim_A2B"] = float(nn_sim.mean())
            stats["cross_nn_std_sim_A2B"] = float(nn_sim.std())
            # if aligned and same lengths, compute match rate
            if n_a == n_b:
                match_rate = float((nn_idx == idx).mean())
                stats["cross_nn_match_rate_A2B"] = match_rate
            # symmetric direction
            sample_size_b = min(int(sample_n), n_b)
            idx_b = np.asarray(rng.sample(range(n_b), sample_size_b)) if sample_size_b < n_b else np.arange(n_b)
            sims_b = unit_b[idx_b] @ unit_a.T
            nn_sim_b = sims_b.max(axis=1)
            nn_idx_b = sims_b.argmax(axis=1)
            stats["cross_nn_mean_sim_B2A"] = float(nn_sim_b.mean())
            stats["cross_nn_std_sim_B2A"] = float(nn_sim_b.std())
            if n_a == n_b:
                stats["cross_nn_match_rate_B2A"] = float((nn_idx_b == idx_b).mean())

            results[(a, b)] = stats
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="retrieval.yaml")
def main_hydra(config) -> None:
    # Defer heavy imports to runtime
    import torch
    from experiments.retrieval_faiss import embed_all_modalities, prepare_retrieval_data
    from matbind.data.datamodules.matbind import MatBindDataModule
    from matbind.model.lightning_module import MatBindModule
    from matbind.utils.pylogger import get_pylogger

    LOGGER = get_pylogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    datamodule: MatBindDataModule = hydra.utils.instantiate(config.data)
    datamodule.setup()

    LOGGER.info(f"Loading model from checkpoint: {config.ckpt_path}")

    checkpoint = torch.load(config.ckpt_path, map_location=device, weights_only=True)

    model: MatBindModule = hydra.utils.instantiate(
        config.model,
        world_size=1,
    )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    LOGGER.info("Model weights loaded successfully")

    # Always use float32 for inference to avoid dtype issues
    LOGGER.info("Converting model to float32 and moving to device")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    # Fix encoder dtype attributes (some encoders store self.dtype)
    LOGGER.info("Scanning for modules with dtype attributes that need updating...")
    dtype_modules_found = 0
    for name, module in model.named_modules():
        if hasattr(module, 'dtype') and module.dtype is not None:
            LOGGER.info(f"  Found module with dtype: {name}, dtype={module.dtype}")
            if module.dtype == torch.bfloat16:
                LOGGER.info(f"  - Changing {name}.dtype from bfloat16 to float32")
                module.dtype = torch.float32
                dtype_modules_found += 1

    if dtype_modules_found > 0:
        LOGGER.info(f"Updated {dtype_modules_found} encoder dtype attributes")
    else:
        LOGGER.info("No encoder dtype attributes needed updating")

    # Also ensure all parameters are actually float32
    non_float32_params = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            LOGGER.warning(f"  Parameter {name} still has dtype {param.dtype}, converting...")
            param.data = param.data.float()
            non_float32_params += 1

    if non_float32_params > 0:
        LOGGER.info(f"Converted {non_float32_params} non-float32 parameters")

    # Verify final dtype
    final_param = next(model.parameters())
    LOGGER.info(f"Final model dtype: {final_param.dtype}, device: {final_param.device}")
    LOGGER.info("Model ready for inference")

    embeddings_dict = embed_all_modalities(
        model=model,
        datamodule=datamodule,
        modalities=config.data.modalities,
        batch_size=config.data.get("batch_size", 128),
        device=device,
    )
    embeddings_for_faiss, aligned_df = prepare_retrieval_data(
        embeddings_dict=embeddings_dict,
        val_data=datamodule.val_data,
        central_modality=config.data.get("central_modality", None),
    )
    pair_samples = getattr(config, "pair_samples", 20000)
    nn_sample = getattr(config, "nn_sample", 1000)
    plot_prefix = getattr(config, "plot_prefix", None)

    # analyze all modalities generated
    for analysis_mod, emb_tensor in embeddings_for_faiss.items():
        try:
            if hasattr(emb_tensor, "numel") and int(emb_tensor.numel()) == 0:
                LOGGER.info(f"Skipping empty embeddings for modality '{analysis_mod}'")
                continue
        except Exception:
            pass

        emb_np = emb_tensor.cpu().numpy() if hasattr(emb_tensor, "cpu") else np.asarray(emb_tensor)
        LOGGER.info(f"Analyzing modality '{analysis_mod}' with embeddings shape {getattr(emb_np, 'shape', None)}")
        metrics = analyze_numpy_embeddings(emb_np, pair_samples=pair_samples, nn_sample=nn_sample, plot_prefix=(None if plot_prefix is None else f"{plot_prefix}_{analysis_mod}"))

    results = compute_modality_gap(embeddings_for_faiss)
    for (mod_a, mod_b), stats in results.items():
        if 'paired_mean' in stats:
            print(f"{mod_a}-{mod_b}:")
            print(f"  Alignment: {stats['paired_mean']:.3f} ± {stats['paired_std']:.3f}")
            print(f"  Retrieval: {stats.get('cross_nn_match_rate_A2B', 'N/A'):.3f}")
            print(f"  Centroid gap: {1 - stats['centroid_cosine']:.3f}")
    
if __name__ == "__main__":
    # No embeddings file passed: run Hydra flow to generate embeddings and analyze
    main_hydra()
