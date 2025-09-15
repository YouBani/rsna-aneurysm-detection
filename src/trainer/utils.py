import os
import random
import json
from pathlib import Path
from typing import Optional, Any

import torch
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pos_weight_from_jsonl(jsonl_path: str) -> torch.Tensor:
    """
    Calculate the positive class weight for BCEWithLogitsLoss.
    The formula is neg_count / pos_count.
    """
    labels = [
        int(json.loads(l)["label"]) for l in Path(jsonl_path).read_text().splitlines()
    ]
    pos = sum(labels)
    neg = len(labels) - pos
    weight = neg / max(pos, 1)
    return torch.tensor([weight], dtype=torch.float32)


def build_binary_metrics():
    return MetricCollection(
        {
            "acc": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "ap": BinaryAveragePrecision(),
        }
    )


def save_checkpoint(
    path: Path,
    model_state: dict,
    optimizer_state: dict,
    epoch: Optional[int] = None,
) -> None:
    dest = Path(path)

    payload = {"model": model_state}
    if optimizer_state:
        payload["optimizer"] = optimizer_state
    if epoch is not None:
        payload["epoch"] = epoch

    tmp = (
        dest.with_suffix(dest.suffix + ".tmp")
        if dest.suffix
        else dest.with_suffix(".tmp")
    )
    torch.save(payload, tmp)
    os.replace(tmp, dest)


def log_metrics(
    logger: Optional[Any], payload: dict[str, Any], step: Optional[int] = None
) -> None:
    """Log metrics if a logger with '.log()' is provided (e.g. wandb)."""
    if logger is None:
        return
    log_fn = getattr(logger, "log", None)
    if not callable(log_fn):
        return
    try:
        if step is None:
            log_fn(payload)
        else:
            log_fn(payload, step=int(step))
    except Exception as e:
        print(f"[log] warning: {e}")


def cuda_reset_peaks(device: torch.device) -> None:
    """Reset CUDA peak memory counters. No-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_empty_cache():
    """Release cached blocks from the CUDA caching allocator back to the driver. No-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_peak_gib(device: torch.device) -> tuple[float, float]:
    """Return (peak_alloc_GiB, peak_reserved_GiB) since last reset.(0.0, 0.0) on CPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return float(alloc), float(reserved)
