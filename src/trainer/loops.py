import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchmetrics import MetricCollection
from tqdm import tqdm
from typing import Optional


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    epoch: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    scaler: Optional[GradScaler] = None,
    accum_steps: int = 1,
    amp_enabled: bool = True,
    empty_cache_every: int = 50,
) -> dict[str, float]:
    if len(loader) == 0:
        raise ValueError("Empty training DataLoader passed to train_one_epoch.")

    model.train()
    metrics = metrics.to("cpu")
    metrics.reset()

    total_loss = 0.0
    total_items = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(loader, desc=f"Epoch{epoch} Train,", unit="batch")
    for step, batch in enumerate(progress_bar):
        x = batch["image"].to(device, non_blocking=True)  # (B, 1, Z, H, W)
        y = batch["label"].to(device, non_blocking=True)  # (B,)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x).squeeze(1)  # (B,)
            loss = loss_fn(logits, y) / accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = torch.sigmoid(logits).detach().float().cpu()
            labels = (y > 0.5).int().cpu()
            metrics.update(preds, labels)

            bs = x.size(0)
            total_loss += float(loss.detach().cpu()) * bs * accum_steps
            total_items += bs

            progress_bar.set_postfix(loss=loss.item() * accum_steps)

        del x, y, logits, preds, labels, loss
        if (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

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
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool = False,
    empty_cache_every: int = 50,
) -> dict[str, float]:
    model.eval()
    metrics = metrics.to("cpu")
    metrics.reset()

    total_loss = 0.0
    total_items = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Val", unit="batch")
    for step, batch in enumerate(progress_bar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

        preds = torch.sigmoid(logits).detach().float().cpu()

        # ------- TEMPORARY REMOVE !!!! ---
        if step == 0:
            print("[val] sample preds:", preds[:10])
            print(
                f"[val] sample preds: mean={preds.mean():.3f} min={preds.min():.3f} max={preds.max():.3f}"
            )

        labels = (y > 0.5).int().cpu()
        metrics.update(preds, labels)

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_items += bs

        progress_bar.set_postfix(loss=loss.item())

        del x, y, logits, preds, labels, loss
        if (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_items
    computed = metrics.compute()
    metrics_dict = {f"val/{k}": v.item() for k, v in computed.items()}
    metrics_dict["val/loss"] = avg_loss
    metrics_dict["epoch"] = epoch
    return metrics_dict
