import hashlib
import pickle
from pathlib import Path
from pprint import pformat

import hydra
import numpy as np
from omegaconf import DictConfig

from matbind.utils.pylogger import get_pylogger
from matbind.data.datamodules.matbind import MatBindDataModule
from types import SimpleNamespace

LOGGER = get_pylogger(__name__)


def _array_hash(arr: np.ndarray) -> str:
    # stable hash for numpy arrays: include shape and dtype
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def extract_crystal_repr(data) -> dict:
    """Extract a compact, hashable representation from a crystal Data object."""
    parts = {}
    # x, edge_index, edge_attr expected
    if hasattr(data, "x") and data.x is not None:
        x = data.x.cpu().numpy()
        parts["x_hash"] = _array_hash(x)
        parts["x_shape"] = x.shape
    else:
        parts["x_hash"] = None

    if hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index.cpu().numpy()
        parts["edge_index_hash"] = _array_hash(ei)
        parts["edge_index_shape"] = ei.shape
    else:
        parts["edge_index_hash"] = None

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        ea = data.edge_attr.cpu().numpy()
        parts["edge_attr_hash"] = _array_hash(ea)
        parts["edge_attr_shape"] = ea.shape
    else:
        parts["edge_attr_hash"] = None

    return parts


def _find_central_in_nested(batch_obj, central_key: str):
    """Recursively search for the central modality batch inside a nested CombinedLoader output.

    Returns the found object or None.
    """
    if isinstance(batch_obj, dict):
        if central_key in batch_obj:
            return batch_obj[central_key]
        for v in batch_obj.values():
            found = _find_central_in_nested(v, central_key)
            if found is not None:
                return found
        return None

    if isinstance(batch_obj, (list, tuple)):
        for el in batch_obj:
            found = _find_central_in_nested(el, central_key)
            if found is not None:
                return found
        return None

    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="retrieval.yaml")
def main(config: DictConfig) -> None:
    LOGGER.info("Instantiating datamodule from config.data...")
    datamodule: MatBindDataModule = hydra.utils.instantiate(config.data)
    # attach a minimal fake trainer so build_dataloader can construct DistributedSampler
    datamodule.trainer = SimpleNamespace(strategy=SimpleNamespace(distributed_sampler_kwargs={"rank": 0, "num_replicas": 1}))
    datamodule.setup()

    central = config.data.central_modality

    central = config.data.central_modality
    # modalities = list(config.data.modalities)
    # if central in modalities:
    #     modalities.remove(central)

    # # For each modality pair (central, modality), build sanitized dataset and extract per-mid repr
    # mid_to_entries: dict[str, list[tuple[str, dict]]] = {}

    # for modality in modalities:
    #     modality_pair = (central, modality)
    #     LOGGER.info(f"Processing modality pair: {modality_pair}")
    #     sanitized = datamodule.dataset_builder.get_modality_data(datamodule.val_data, modality_pair)
    #     datasets = datamodule.dataset_builder.build_datasets(sanitized, modality_pair)

    #     central_dataset = datasets[central]
    #     # iterate and collect mid->repr
    #     for idx in range(len(central_dataset)):
    #         item = central_dataset[idx]
    #         mid = getattr(item, "mid", None)
    #         mid = str(mid)
    #         reprd = extract_crystal_repr(item)
    #         mid_to_entries.setdefault(mid, []).append((modality, reprd))

    # # Now check mids that appear in multiple modality-pairs
    # mismatches = {}
    # for mid, entries in mid_to_entries.items():
    #     if len(entries) <= 1:
    #         continue
    #     # compare all entries' hashes
    #     hashes = [(mod, (e.get("x_hash"), e.get("edge_index_hash"), e.get("edge_attr_hash"))) for mod, e in entries]
    #     unique = {h for _, h in hashes}
    #     if len(unique) > 1:
    #         mismatches[mid] = {
    #             "count": len(entries),
    #             "per_modality": dict(hashes),
    #             "details": entries,
    #         }

    # if mismatches:

    #     LOGGER.warning(f"Found {len(mismatches)} mids with differing crystal representations across modality pairs.")
    #     # Print sample
    #     sample = list(mismatches.items())[:10]
    #     for mid, info in sample:
    #         LOGGER.warning(f"MID: {mid}, occurrences: {info['count']}")
    #         LOGGER.warning(f"Per-modality hashes: {pformat(info['per_modality'])}")
    #     return

    # LOGGER.info("No mismatching crystal representations found across modality-pairs. All identical for equal material ids.")
    # --- Now check actual val_dataloader behavior (batches) ---
    LOGGER.info("Now checking validation dataloader delivery for central modality consistency across modality-pairs...")
    _, val_loader = datamodule.build_dataloader(
        data = datamodule.val_data,
        shuffle=False,
        drop_last=False,
        combined_loader_mode="sequential",
    )
    batch_limit = int(config.get("batch_check_max_batches", 20))

    dl_mid_to_entries: dict[str, list[tuple[str, int, dict]]] = {}

    for batch, batch_idx, dataloader_idx in val_loader:
        if isinstance(batch, (list, tuple)):
            # CombinedLoader sometimes yields a tuple like (batch_dict, dataloader_idx)
            # find the first dict-like element to iterate modality keys
            batch_map = None
            for el in batch:
                if isinstance(el, dict):
                    batch_map = el
                    break
            if batch_map is None:
                # fallback: try to find central modality anywhere in the tuple
                central_batch = _find_central_in_nested(batch, central)
                if central_batch is None:
                    continue
                outer_items = [("<top>", batch)]
            else:
                outer_items = batch_map.items()
        elif isinstance(batch, dict):
            outer_items = batch.items()
        else:
            # unknown structure: try to find central directly
            central_batch = _find_central_in_nested(batch, central)
            if central_batch is None:
                continue
            outer_items = [("<top>", batch)]

        for outer_modality, nested in outer_items:
            # find the central modality batch inside nested
            central_batch = _find_central_in_nested(nested, central)
            if central_batch is None:
                continue

            # central_batch may be:
            # - a torch_geometric.data.Batch (has to_data_list)
            # - an IndexedSubBatch tuple like (batch_obj, ...) where batch_obj is a Batch
            # - a list/tuple of Data objects
            # Normalize to a list of Data objects
            if isinstance(central_batch, (list, tuple)):
                # If it's an IndexedSubBatch tuple, the first element is the batch object
                first = central_batch[0]
                if hasattr(first, "to_data_list"):
                    data_list = first.to_data_list()
                elif isinstance(first, (list, tuple)):
                    data_list = list(first)
                else:
                    data_list = [first]
            elif hasattr(central_batch, "to_data_list"):
                data_list = central_batch.to_data_list()
            else:
                # single item
                data_list = [central_batch]

            for item in data_list:
                mid = getattr(item, "mid", None)
                mid = str(mid)
                reprd = extract_crystal_repr(item)
                dl_mid_to_entries.setdefault(mid, []).append((outer_modality, batch_idx, reprd))

    # Now find mids that appear multiple times across dataloader deliveries and compare
    dl_mismatches = {}
    for mid, entries in dl_mid_to_entries.items():
        if len(entries) <= 1:
            continue
        hashes = [(mod, (e.get("x_hash"), e.get("edge_index_hash"), e.get("edge_attr_hash"), idx)) for mod, idx, e in entries]
        unique = {h for _, h in hashes}
        if len(unique) > 1:
            dl_mismatches[mid] = {
                "count": len(entries),
                "per_modality": {mod: (h, idx) for mod, (h, idx) in [(m, (e.get("x_hash"), i)) for m, i, e in entries]},
                "details": entries,
            }

    if not dl_mismatches:
        LOGGER.info(f"No mismatches found in the first {batch_limit} validation dataloader batches.")
    else:
        LOGGER.warning(f"Found {len(dl_mismatches)} mids with differing central representations across validation dataloader deliveries (first {batch_limit} batches).")
        sample = list(dl_mismatches.items())[:10]
        for mid, info in sample:
            LOGGER.warning(f"MID: {mid}, occurrences: {info['count']}")
            LOGGER.warning(f"Sample details: {pformat(info['details'])}")


if __name__ == "__main__":
    main()
