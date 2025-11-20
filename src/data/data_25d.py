import torch
import numpy as np
import random
from torch.utils.data import DataLoader


from .rsna_slice_dataset import RSNASliceDataset, get_transform

__all__ = ["build_loaders"]


def _seed_worker(_workder_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    train_csv: str,
    val_csv: str,
    *,
    image_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[Dataloader, Dataloader]:
    """Build train/val DataLoaders for the 2.5D slice model."""
    train_transform = get_transform(image_size=image_size, is_train=True)
    val_transform = get_transform(image_size=image_size, is_train=False)

    train_ds = RSNASliceDataset(manifest_path=train_csv, transform=train_transform)
    val_ds = RSNASliceDataset(manifest_path=val_csv, transform=val_transform)

    g = torch.Generator().manual_seed(seed)

    kwargs = {}
    if pin_memory:
        kwargs["pin_memory"] = True
        kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=True,
        **kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        drop_last=False,
        **kwargs,
    )

    return train_loader, val_loader
