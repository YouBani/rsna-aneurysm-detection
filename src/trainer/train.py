from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from src.trainer.loops import train_one_epoch, validate
from src.trainer.utils import (
    build_binary_metrics,
    seed_all,
    save_checkpoint,
)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    seed: int,
    checkpoint_dir: str,
    scaler: Optional[GradScaler] = None,
    accum_steps: int = 1,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    empty_cache_every: int = 50,
) -> dict[str, Any]:
    seed_all(seed)

    base = build_binary_metrics().to(device)
    train_metrics = base.clone()
    val_metrics = base.clone()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"
    best_auroc = float("-inf")

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader, "sampler") and hasattr(
            train_loader.sampler, "generator"
        ):
            train_loader.sampler.generator.manual_seed(seed + epoch)

        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=train_metrics,
            epoch=epoch,
            device=device,
            scaler=scaler,
            accum_steps=accum_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            empty_cache_every=empty_cache_every,
        )

        # free fragmentation between phases
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        val_out = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            metrics=val_metrics,
            epoch=epoch,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            empty_cache_every=empty_cache_every,
        )

        t_loss = train_out.get("train/loss", float("nan"))
        t_acc = train_out.get("train/acc", float("nan"))
        t_auc = train_out.get("train/auroc", float("nan"))

        v_loss = val_out.get("val/loss", float("nan"))
        v_acc = val_out.get("val/acc", float("nan"))
        v_auc = val_out.get("val/auroc", float("nan"))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train: loss={t_loss:.4f}, acc={t_acc:.4f}, auroc={t_auc:.4f} | "
            f"val:   loss={v_loss:.4f}, acc={v_acc:.4f}, auroc={v_auc:.4f}"
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
