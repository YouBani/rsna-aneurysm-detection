import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchmetrics import MetricCollection

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from typing import Optional

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    epoch: int,
    device: str,
    scaler: Optional[GradScaler | None] = None,
) -> dict[str, float]:
    """
    """
    if len(loader) == 0:
        raise ValueError("Empty training DataLoader passed to train_one_epoch.")

    model.train()
    metrics = metrics.to(device)
    metrics.reset()

    total_loss = 0.0
    total_items = 0.0

    for step, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True) # (B, 1, Z, H, W)
        y = batch["label"].to(device, non_blocking=True) # (B,)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            logits = model(x).squeeze(1)                 # (B,)
            loss = loss_fn(logits, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            metrics.update(preds, (y > 0.5).int())

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_items += bs

    avg_loss = total_loss / total_items
    computed = metrics.compute()
    metrics_dict = {f"train/{k}": v.item() for k, v in computed.items()}
    metrics_dict.update({"train/loss": avg_loss, "epoch": epoch})

    return metrics_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    epoch: int,
    device: str,
) -> dict[str, float]:
    """
    """
    model.eval()
    metrics = metrics.to(device)
    metrics.reset()

    total_loss = 0.0
    total_items = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)

        preds = torch.sigmoid(logits)
        metrics.update(preds, (y > 0.5).int())

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_items += bs

    avg_loss = total_loss / total_items
    computed = metrics.compute()
    metrics_dict = {f"val/{k}": v.item() for k, v in computed.items()}
    metrics_dict["val/loss"] = avg_loss
    metrics_dict["epoch"] = epoch

    return metrics_dict
