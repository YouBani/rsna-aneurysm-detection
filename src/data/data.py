import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import RSNADataset

__all__ = ["build_loaders"]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _labels_from_jsonl(jsonl_path: str) -> list[int]:
    with open(jsonl_path) as f:
        return [int(json.loads(l)["label"]) for l in f]


def build_loaders(
    train_jsonl: str,
    val_jsonl: str,
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_dir: str | None = None,
    target_slices: int = 128,
    seed: int = 42,
    weighted_sampling: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader, RSNADataset, RSNADataset]:
    """
    Build train/val DataLoaders for 3D volumes coming from RSNADataset.
    Returns: train_loader, val_loader, train_ds, val_ds
    """
    train_ds = RSNADataset(train_jsonl, target_slices=target_slices, cache_dir=cache_dir)
    val_ds = RSNADataset(val_jsonl, target_slices=target_slices, cache_dir=cache_dir)

    g = torch.Generator().manual_seed(42)

    sampler = None
    shuffle = True
    if weighted_sampling:
        labels = _labels_from_jsonl(train_jsonl)
        pos = sum(labels)
        neg = len(labels) - pos
        weights = {0: 1.0 / max(neg, 1), 1: 1.0 / max(pos, 1)}
        sample_weights = [weights[y] for y in labels]
        
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True, generator=g)
        shuffle = False 

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return train_loader, val_loader, train_ds, val_ds