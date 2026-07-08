#!/usr/bin/env python3
"""
Compare two aggregated embeddings pickle files created by `experiments/retrieval.py`.
Usage:
  python scripts/compare_embeddings.py /path/to/emb_with_groupna.pkl /path/to/emb_without_groupna.pkl --modalities dos,pxrd --tol 1e-6

The script prints per-modality statistics: number of common mids, exact matches, within tolerance, and the top mismatches by L2 distance.
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_embeddings(p: Path) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    with open(p, "rb") as f:
        data = pickle.load(f)
    out = {}
    for modality, value in data.items():
        if isinstance(value, tuple) and len(value) >= 2:
            tensor, mids = value[0], value[1]
            # support torch tensors
            try:
                import torch

                if hasattr(tensor, "detach"):
                    arr = tensor.detach().cpu().numpy()
                else:
                    arr = np.asarray(tensor)
            except Exception:
                arr = np.asarray(tensor)
            mids = [str(m).strip() for m in mids]
            out[modality] = (arr, mids)
        else:
            raise ValueError(f"Unsupported embedding format for modality {modality}")
    return out


def l2_and_cosine(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = a - b
    l2 = np.linalg.norm(diff, axis=1)
    # cosine similarity
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm * b_norm, 1e-12)
    cos = np.sum(a * b, axis=1) / denom
    return l2, cos


def compare_modality(
    mod: str,
    emb_a: Tuple[np.ndarray, List[str]],
    emb_b: Tuple[np.ndarray, List[str]],
    tol: float = 1e-6,
    topk: int = 10,
):
    arr_a, mids_a = emb_a
    arr_b, mids_b = emb_b
    # build pos maps
    pos_a = {mid: i for i, mid in enumerate(mids_a)}
    pos_b = {mid: i for i, mid in enumerate(mids_b)}

    common = sorted(set(mids_a) & set(mids_b))
    if not common:
        print(f"Modality {mod}: no common mids between files")
        return

    idx_a = [pos_a[m] for m in common]
    idx_b = [pos_b[m] for m in common]

    sub_a = arr_a[idx_a]
    sub_b = arr_b[idx_b]

    if sub_a.shape != sub_b.shape:
        print(f"Modality {mod}: shape mismatch after indexing: {sub_a.shape} vs {sub_b.shape}")
        return

    l2, cos = l2_and_cosine(sub_a, sub_b)
    exact_eq = np.sum(l2 == 0.0)
    within_tol = np.sum(l2 <= tol)
    total = len(common)

    print(f"\nModality: {mod}")
    print(f"  common mids: {total}")
    print(f"  exact equal (L2==0): {exact_eq}")
    print(f"  within tol (L2<={tol}): {within_tol}")
    print(f"  max L2: {float(np.max(l2)):.6e}")
    print(f"  min L2: {float(np.min(l2)):.6e}")
    print(f"  mean L2: {float(np.mean(l2)):.6e}")
    print(f"  mean cosine: {float(np.mean(cos)):.6e}")

    # show top mismatches
    worst_idx = np.argsort(-l2)[:topk]
    print("  Top mismatches (mid, L2, cosine):")
    for i in worst_idx:
        print(f"    {common[i]}  {l2[i]:.6e}  {cos[i]:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", type=str, default=None, help="Comma-separated list of modalities to compare. Default: all common modalities")
    parser.add_argument("--tol", type=float, default=1e-6, help="L2 tolerance to consider embeddings equal")
    parser.add_argument("--topk", type=int, default=10, help="How many top mismatches to show")
    args = parser.parse_args()

    # emb_a = load_embeddings("/p/project1/solai/yang21/MatBind/experiments/embeddings_new_scripts.pkl")
    with open("/p/project1/solai/yang21/MatBind/experiments/embeddings_without_dropna.pkl", "rb") as f:
        data = pickle.load(f)
    print(data["crystal_structure__dup_by_source"])

    # if args.modalities:
    #     mods = [m.strip() for m in args.modalities.split(",")]
    # else:
    #     mods = sorted(set(emb_a.keys()) & set(emb_b.keys()))

    # if not mods:
    #     print("No common modalities found between the two files.")
    #     return

    # for mod in mods:
    #     if mod not in emb_a:
    #         print(f"Modality {mod} missing from first file")
    #         continue
    #     if mod not in emb_b:
    #         print(f"Modality {mod} missing from second file")
    #         continue
    #     compare_modality(mod, emb_a[mod], emb_b[mod], tol=args.tol, topk=args.topk)


if __name__ == "__main__":
    main()
