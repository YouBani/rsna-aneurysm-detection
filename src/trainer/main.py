import argparse
from pathlib import Path
import wandb

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp.grad_scaler import GradScaler
from src.data import build_loaders
from src.models.model import build_3d_model
from src.trainer.train import train
from src.trainer.utils import seed_all, pos_weight_from_jsonl_multilabel
from src.constants.rsna import K


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.jsonl")
    p.add_argument("--val", required=True, help="Path to val.jsonl")
    p.add_argument("--out", required=True, help="Checkpoint directory")
    p.add_argument(
        "--cache", default=None, help="Cache dir for preprocessed volumes (.npy)"
    )
    p.add_argument("--target_slices", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Set training precision: fp32, fp16 (mixed), bf16 (mixed)",
    )
    p.add_argument("--ckpt", action="store_true", help="Enable gradient checkpointing")
    p.add_argument(
        "--empty_cache_every",
        type=int,
        default=50,
        help="Call torch.cuda.empty_cache() every N steps (0 to disable)",
    )
    p.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )
    p.add_argument(
        "--wandb",
        default="off",
        choices=["off", "online", "offline"],
        help="Weights & Biases logging mode",
    )
    p.add_argument("--run_name", default=None, help="Optional W&B run name")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    cfg = dict(
        model="r3d_18",
        use_groupnorm=False,
        precision=args.precision,
        bs=args.bs,
        accum_steps=args.accum_steps,
        lr=args.lr,
        wd=args.wd,
        target_slices=args.target_slices,
        weighted_sampling=False,
    )

    run = None
    if args.wandb != "off":
        run = wandb.init(
            project="rsna-aneurysm",
            mode=args.wandb,
            name=args.run_name,
            config=cfg,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, *_ = build_loaders(
        train_jsonl=args.train,
        val_jsonl=args.val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        cache_dir=args.cache,
        target_slices=args.target_slices,
        seed=args.seed,
        weighted_sampling=False,
    )

    # Model
    model = build_3d_model(
        in_channels=1, num_outputs=K, checkpointing=args.ckpt, use_groupnorm=False
    ).to(device)

    # Loss (class imbalance from TRAIN jsonl)
    pos_w = pos_weight_from_jsonl_multilabel(args.train).to(device)
    pos_w = pos_w.clamp_(1.0, 10.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        cooldown=0,
        min_lr=1e-6,
    )

    if args.precision == "bf16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
        scaler = GradScaler(enabled=False)
    elif args.precision == "fp16":
        amp_enabled = True
        amp_dtype = torch.float16
        scaler = GradScaler(enabled=True)
    else:
        amp_enabled = False
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_dir=args.out,
        scaler=scaler,
        accum_steps=args.accum_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        empty_cache_every=args.empty_cache_every,
        logger=run,
        scheduler=scheduler,
    )

    print(summary)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
