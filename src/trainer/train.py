import time
import math
import numpy as np
from pathlib import Path
from typing import Optional, Any

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.trainer.loops import train_one_epoch, validate
from src.trainer.act_hooks import ActivationHook
from src.trainer.utils import (
    build_binary_metrics,
    seed_all,
    save_checkpoint,
    log_metrics,
    cuda_peak_gib,
    cuda_empty_cache,
    cuda_reset_peaks,
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
    logger: Optional[Any] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
    act_hook: bool = False,
    act_layers: tuple[str, ...] = ("stem", "layer1", "layer2", "layer3", "layer4"),
    act_sample_per_call: int = 256,
    act_every_n: int = 5,
    act_hist_every: int = 2,
) -> dict[str, Any]:
    seed_all(seed)

    base = build_binary_metrics().to(device)
    train_metrics = base.clone()
    val_metrics = base.clone()

    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir_path / "best_model.pth"
    last_path = checkpoint_dir_path / "last_model.pth"
    best_final = float("-inf")

    ah: Optional[ActivationHook] = None
    if act_hook:
        ah = ActivationHook(
            model=model,
            layers_to_hook=act_layers,
            sample_per_call=act_sample_per_call,
            every_n=act_every_n,
        )

    ctx = ah if (act_hook and ah is not None) else nullcontext()
    with ctx:
        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "generator"
            ):
                train_loader.sampler.generator.manual_seed(seed + epoch)  # type: ignore

            cuda_reset_peaks(device)
            t0 = time.perf_counter()

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

            t1 = time.perf_counter()

            steps_per_sec = len(train_loader) / max(t1 - t0, 1e-9)
            peak_alloc, peak_resv = cuda_peak_gib(device)
            log_metrics(
                logger,
                {
                    "perf/steps_per_sec": steps_per_sec,
                    "cuda/peak_alloc_GiB": peak_alloc,
                    "cuda/peak_reserved_GiB": peak_resv,
                },
                step=epoch,
            )

            cuda_empty_cache()
            cuda_reset_peaks(device)

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

            v_alloc, v_resv = cuda_peak_gib(device)

            log_metrics(
                logger,
                {
                    "cuda/val_peak_alloc_GiB": v_alloc,
                    "cuda/val_peak_reserved_GiB": v_resv,
                },
                step=epoch,
            )

            # Means every epoch
            if act_hook and ah is not None:
                act_means = ah.summarize_means()
                log_metrics(logger, act_means, step=epoch)

            # Hists every K epochs
            if (
                act_hook
                and ah is not None
                and act_hist_every > 0
                and (epoch % act_hist_every == 0)
            ):
                act_hists = ah.summarize_hists()
                if act_hists:
                    try:
                        import wandb

                        wb_hists = {}
                        pct = {}
                        qs = [50, 90, 99]
                        for k, v in act_hists.items():
                            arr = np.asarray(v, dtype=float).reshape(-1)
                            wb_hists[f"hists/{k}"] = wandb.Histogram(arr.tolist())

                            p = np.percentile(arr, qs)
                            for q, v in zip(qs, p):
                                pct[f"debug/act_pct/{k}/p{q}"] = float(v)

                        log_metrics(logger, {**wb_hists, **pct}, step=epoch)
                    except Exception:
                        hist_summary = {}
                        qs = [0, 1, 25, 50, 75, 90, 100]
                        for k, arr in act_hists.items():
                            pct = np.percentile(arr, qs)
                            for q, v in zip(qs, pct):
                                hist_summary[f"{k}/p{q}"] = float(v)
                            log_metrics(logger, hist_summary, step=epoch)

            t_loss = train_out.get("train/loss", float("nan"))
            t_acc = train_out.get("train/acc", float("nan"))
            t_auc = train_out.get("train/auroc", float("nan"))

            v_loss = val_out.get("val/loss", float("nan"))
            v_acc = val_out.get("val/acc", float("nan"))
            v_auc_p = val_out.get("val/auc_present", float("nan"))
            v_final = val_out.get("val/final_auc_weighted", float("nan"))
            v_locavg = val_out.get("val/auc_locations_mean", float("nan"))

            if scheduler is not None:
                scheduler.step(v_final)

            current_lr = optimizer.param_groups[0]["lr"]

            log_metrics(
                logger,
                {
                    "train/loss": t_loss,
                    "train/acc": t_acc,
                    "train/auroc": t_auc,
                    "val/loss": v_loss,
                    "val/acc": v_acc,
                    "val/auc_present": v_auc_p,
                    "val/auc_locations_mean": v_locavg,
                    "val/final_auc_weighted": v_final,
                    "lr": current_lr,
                    "epoch": epoch,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train: loss={t_loss:.4f}, acc={t_acc:.4f}, auroc={t_auc:.4f} | "
                f"val:   loss={v_loss:.4f}, acc={v_acc:.4f}, auc_present={v_auc_p:.4f}, final={v_final:.4f}"
            )

            save_checkpoint(
                last_path,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=(scheduler.state_dict() if scheduler else None),
                epoch=epoch,
            )

            if not math.isnan(v_final) and v_final > best_final:
                best_final = v_final
                save_checkpoint(
                    best_path,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=(scheduler.state_dict() if scheduler else None),
                    epoch=epoch,
                )
                print(f"New best FINAL={best_final:.4f} - saved {best_final}")
                log_metrics(
                    logger, {"checkpoint/best_final_auc": best_final}, step=epoch
                )

    return {
        "final_auc_weighted": val_out.get("val/final_auc_weighted", float("nan")),
        "auc_present": val_out.get("val/auc_present", float("nan")),
        "best_model_path": best_path,
        "last_model_path": last_path,
    }
