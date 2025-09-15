import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import RSNADataset

__all__ = ["build_loaders"]


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _labels_from_jsonl(jsonl_path: str) -> list[int]:
    with open(jsonl_path) as f:
        return [int(json.loads(line)["label"]) for line in f]


def build_loaders(
    train_jsonl: str,
    val_jsonl: str,
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    target_slices: int = 128,
    seed: int = 42,
    weighted_sampling: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, RSNADataset, RSNADataset]:
    """
    Build train/val DataLoaders for 3D volumes coming from RSNADataset.
    Returns: train_loader, val_loader, train_ds, val_ds
    """
    train_ds = RSNADataset(
        train_jsonl, target_slices=target_slices, cache_dir=cache_dir
    )
    val_ds = RSNADataset(val_jsonl, target_slices=target_slices, cache_dir=cache_dir)

    g = torch.Generator().manual_seed(seed)

    kwargs = {}
    if pin_memory:
        kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    sampler = None
    shuffle = True
    if weighted_sampling:
        labels = _labels_from_jsonl(train_jsonl)
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
        **kwargs,
    )

    return train_loader, val_loader, train_ds, val_ds
