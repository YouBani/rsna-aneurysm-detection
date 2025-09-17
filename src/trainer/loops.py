import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchmetrics import MetricCollection
from tqdm import tqdm
from typing import Optional

from src.metrics.rsna import build_multilabel_auroc, weighted_multilabel_auc
from src.constants.rsna import PRESENT_IDX


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
        y14 = batch["labels14"].to(device, non_blocking=True).float()  # (B, 14)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits14 = model(x)  # (B, 14)
            loss = loss_fn(logits14, y14) / accum_steps

        assert logits14.shape[-1] == 14, f"Expected 14 outputs, got {logits14.shape}"

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
            preds = torch.sigmoid(logits14[:, PRESENT_IDX]).detach().float().cpu()
            labels_present = (y14[:, PRESENT_IDX] > 0.5).int().cpu()
            metrics.update(preds, labels_present)

            bs = x.size(0)
            total_loss += float(loss.detach().cpu()) * bs * accum_steps
            total_items += bs

            progress_bar.set_postfix(loss=loss.item() * accum_steps)

        del x, y14, logits14, preds, labels_present, loss
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

    ml_auroc = build_multilabel_auroc()

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Val", unit="batch")
    for step, batch in enumerate(progress_bar):
        x = batch["image"].to(device, non_blocking=True)
        y14 = batch["labels14"].to(device, non_blocking=True).float()  # (B, 14)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits14 = model(x)
            loss = loss_fn(logits14, y14)

        assert logits14.shape[-1] == 14, f"Expected 14 outputs, got {logits14.shape}"

        preds_present = torch.sigmoid(logits14[:, PRESENT_IDX]).detach().float().cpu()
        labels_present = (y14[:, PRESENT_IDX] > 0.5).int().cpu()
        metrics.update(preds_present, labels_present)

        scores14 = torch.sigmoid(logits14).detach().float().cpu()
        ml_auroc.update(scores14, y14.detach().int().cpu())

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_items += bs
        progress_bar.set_postfix(loss=loss.item())

        del x, y14, logits14, scores14, preds_present, labels_present, loss
        if (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_items
    metrics_dict = {f"val/{k}": v.item() for k, v in metrics.compute().items()}
    metrics_dict["val/loss"] = avg_loss
    metrics_dict["epoch"] = epoch

    # competition metric
    per_label_auc = ml_auroc.compute()
    metrics_dict["val/final_auc_weighted"] = weighted_multilabel_auc(
        per_label_auc
    ).item()

    present_auc = torch.nan_to_num(per_label_auc[PRESENT_IDX], nan=0.5).item()
    others = torch.cat([per_label_auc[:PRESENT_IDX], per_label_auc[PRESENT_IDX + 1 :]])
    others_mean = torch.nan_to_num(others, nan=0.5).mean().item()
    metrics_dict["val/auc_present"] = present_auc
    metrics_dict["val/auc_locations_mean"] = others_mean

    return metrics_dict
