import json
import random
from typing import Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import RSNADataset

__all__ = ["build_loaders"]


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_jsonl_rows(jsonl_path: str) -> list[dict[str, Any]]:
    """Loads the rows from a .jsonl file."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def build_loaders(
    train_jsonl: str,
    val_jsonl: str,
    preprocessed_dir: str,
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    seed: int = 42,
    weighted_sampling: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, RSNADataset, RSNADataset]:
    """
    Build train/val DataLoaders for preprocessed 3D volumes.
    Returns: train_loader, val_loader, train_ds, val_ds
    """
    train_rows = _load_jsonl_rows(train_jsonl)
    val_rows = _load_jsonl_rows(val_jsonl)

    data_root = Path(preprocessed_dir)
    train_ds = RSNADataset(preprocessed_dir=data_root, manifest_rows=train_rows)
    val_ds = RSNADataset(preprocessed_dir=data_root, manifest_rows=val_rows)

    g = torch.Generator().manual_seed(seed)

    kwargs = {}
    if pin_memory:
        kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    sampler = None
    shuffle = True
    if weighted_sampling:
        labels = [int(row["label"]) for row in train_rows]
        pos = sum(labels)
        neg = len(labels) - pos
        weights = {0: 1.0 / max(neg, 1), 1: 1.0 / max(pos, 1)}
        sample_weights = [weights[y] for y in labels]

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True, generator=g
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        persistent_workers=(num_workers > 0),
        generator=(g if sampler is None else None),
        drop_last=False,
        timeout=120,
        **kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        timeout=120,
        **kwargs,
    )

    return train_loader, val_loader, train_ds, val_ds
