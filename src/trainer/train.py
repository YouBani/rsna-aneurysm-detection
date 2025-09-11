from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from src.data import build_loaders
from src.models.model import build_3d_model
from src.trainer.loops import train_one_epoch, validate
from src.trainer.utils import (
    build_binary_metrics,
    seed_all,
    pos_weight_from_jsonl,
    save_checkpoint,
)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: str,
    epochs: int,
    seed: int,
    checkpoint_dir: str,
    scaler: Optional[GradScaler]=None,
) -> dict[str, Any]:
    seed_all(seed)

    scaler = GradScaler(enabled=scaler is not None)

    base = build_binary_metrics().to(device)
    train_metrics = base.clone()
    val_metrics = base.clone()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"
    best_auroc = 0.0

    for epoch in range(1, epochs + 1):
        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=train_metrics,
            epoch=epoch,
            device=device,
            scaler=scaler,
        )
        val_out = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            metrics=val_metrics,
            epoch=epoch,
            device=device,
        )

        t_acc = train_out.get("train/acc", float("nan"))
        t_auc = train_out.get("train/auroc", float("nan"))
        v_acc = train_out.get("val/acc", float("nan"))
        v_auc = train_out.get("val/auroc", float("nan"))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train: loss={train_out['train/loss']:.4f}, acc={t_acc:.4f}, auroc={t_auc:.4f} | "
            f"val:   loss={val_out['val/loss']:.4f}, acc={v_acc:.4f}, auroc={v_auc:.4f}"
        )

        save_checkpoint(
            last_path,
            model.state_dict(),
            optimizer.state_dict(),
            epoch=epoch,
        )
        current_auroc = val_out["val/auroc"]
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            save_checkpoint(
                best_path,
                model.state_dict(),
                optimizer.state_dict(),
                epoch=epoch,
            )
            print(f"New best AUROC={best_auroc:.4f} - saved {best_auroc}")

    return {
        "auroc": val_out["val/auroc"],
        "ap": val_out["val/ap"],
        "best_model_path": best_path,
        "last_model_path": last_path,
    }
