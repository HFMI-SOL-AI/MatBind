import torch

from matbind.utils import select_device

from matbind.utils.pylogger import get_pylogger
from typing import Any
LOGGER = get_pylogger(__name__)

def aggregate_embeddings(
    embeddings: list,
    modalities: list[str],
    central_modality: str,
) -> dict[str, Any]:
    """
    Aggregate prediction outputs produced by `MatBindModule.prediction_aggregate_batches`.

    Expected `predictions` to be a list (per dataloader) of tuples: (embedding_dict, mids_list),
    where `embedding_dict` maps modality->torch.Tensor for that dataloader and `mids_list` is the
    concatenated list of material ids for the dataloader in the same order as the concatenated tensors.

    Returns a dict mapping modality -> (tensor, mids_list) where mids_list corresponds to the rows
    of the tensor.
    """
    device = select_device()

    # Trainer.predict often returns a list-of-lists (per dataloader) of per-batch tuples.
    # Accept either: list[tuple] or list[list[tuple]] and normalize to a flat list of items.
    flat_items = []
    for x in embeddings:
        if isinstance(x, list):
            flat_items.extend(x)
        else:
            flat_items.append(x)
    LOGGER.debug(f"aggregate_embeddings modalities={modalities}")
    # Normalized container per modality: list of (tensor, mids, source_label) entries
    # source_label is the dataloader/source modality(s) that produced this item (e.g. 'dos' or 'pxrd')
    per_modality: dict[str, list[tuple[torch.Tensor, list[str], str]]] = {m: [] for m in modalities}

    for item in flat_items:
        # item is expected to be (embedding_dict, mids)
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError("Unexpected prediction item format: expected (embedding_dict, mids) or list of such tuples")
        embedding_dict, mids = item
        # normalize mids
        mids = [str(m).strip() for m in mids]
        # Determine a source label for this prediction item: the other modalities present
        keys = list(embedding_dict.keys())
        for modality in modalities:
            if modality in embedding_dict:
                other_keys = [k for k in keys if k != modality]
                source_label = ",".join(sorted(other_keys)) if other_keys else "<solo>"
                per_modality[modality].append((embedding_dict[modality], mids, source_label))

    result: dict[str, Any] = {}
    for modality, items in per_modality.items():
        if not items:
            continue

        tensors = [t for t, _, _ in items]
        mids_lists = [m for _, m, _ in items]
        source_lists = [s for _, _, s in items]

        # concatenate tensors along first dim and flatten mids
        concatenated = torch.cat(tensors, dim=0).to(device)
        flat_mids = [x for mids in mids_lists for x in mids]
        # Build a parallel flat list of source labels aligned with flat_mids
        flat_sources: list[str] = [s for s_list, s in zip(mids_lists, source_lists, strict=True) for s in [s] * len(s_list)]

        # Deduplicate by material id: if a mid appears multiple times for a modality,
        mid_to_positions: dict[str, list[int]] = {}
        for pos, mid in enumerate(flat_mids):
            mid_to_positions.setdefault(mid, []).append(pos)

        dup_mids = [mid for mid, pos_list in mid_to_positions.items() if len(pos_list) > 1]
        if dup_mids:
            sample = dup_mids[:10]
            LOGGER.error(
                f"Found {len(dup_mids)} duplicate material ids for modality '{modality}'. Sample: {sample}. Using first-seen embedding for each duplicate."
            )

            # Diagnostic: check whether duplicate embeddings are identical (within tol)
            tol = 1e-6
            identical_count = 0
            mismatches = []
            for mid in dup_mids:
                positions = mid_to_positions[mid]
                # gather embeddings for this mid
                emb_stack = concatenated[positions]
                # compute L2 distance to the first occurrence
                diffs = torch.norm(emb_stack - emb_stack[0:1], dim=1).cpu().numpy()
                max_l2 = float(diffs.max())
                mean_l2 = float(diffs.mean())
                if max_l2 <= tol:
                    identical_count += 1
                else:
                    mismatches.append((mid, diffs, mean_l2))

            LOGGER.info(
                f"Duplicate embedding diagnostic for modality '{modality}': {identical_count}/{len(dup_mids)} duplicates identical within tol={tol}."
            )
            if mismatches:
                sample_mismatches = mismatches[:10]
                LOGGER.warning(
                    f"Sample mismatching duplicate mids (mid, max_l2, mean_l2): {sample_mismatches}"
                )

            unique_mids: list[str] = []
            unique_embeddings: list[torch.Tensor] = []
            seen: set[str] = set()
            for _, mid in enumerate(flat_mids):
                if mid in seen:
                    continue
                # pick first occurrence position
                first_pos = mid_to_positions[mid][0]
                unique_embeddings.append(concatenated[first_pos].unsqueeze(0))
                unique_mids.append(mid)
                seen.add(mid)

            dedup_tensor = torch.cat(unique_embeddings, dim=0).to(device)
            result[modality] = (dedup_tensor, unique_mids)

            # Additionally, for the central modality save a duplication map keyed by source modality
            if modality == central_modality:
                # Build a map: source_label -> list of dicts with mid and diagnostic info
                dup_by_source: dict[str, list[dict]] = {}
                for mid in dup_mids:
                    positions = mid_to_positions[mid]
                    sources = [flat_sources[pos] for pos in positions]
                    # choose unique source labels seen for this mid
                    unique_sources = sorted(set(sources))
                    # collect L2 diagnostics per position
                    emb_stack = concatenated[positions]
                    diffs = torch.norm(emb_stack - emb_stack[0:1], dim=1).cpu().numpy()
                    entries = []
                    for pos, src, l2 in zip(positions, sources, diffs.tolist(), strict=True):
                        entries.append({"position": int(pos), "source": src, "l2_to_first": float(l2)})

                    for src in unique_sources:
                        dup_by_source.setdefault(src, []).append({"mid": mid, "entries": entries})

                # attach diagnostics under a reserved key in the result dict
                result[f"{modality}__dup_by_source"] = dup_by_source
        else:
            # No duplicates: store concatenated tensor and mids
            result[modality] = (concatenated, flat_mids)

    return result
