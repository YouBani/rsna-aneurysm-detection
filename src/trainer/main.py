import argparse
from pathlib import Path

import torch
from src.data import build_loaders
from src.models.model import build_3d_model
from src.trainer.train import train
from src.trainer.utils import seed_all, pos_weight_from_jsonl

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.jsonl")
    p.add_argument("--val",   required=True, help="Path to val.jsonl")
    p.add_argument("--out",   required=True, help="Checkpoint directory")
    p.add_argument("--cache", default=None,  help="Cache dir for preprocessed volumes (.npy)")
    p.add_argument("--target_slices", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs",     type=int, default=2)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--wd",     type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16/bf16)")
    p.add_argument("--bf16", action="store_true", help="Prefer bfloat16 over fp16 when --amp is used")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, train_ds, _ = build_loaders(
        train_jsonl=args.train,
        val_jsonl=args.val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        cache_dir=args.cache,
        target_slices=args.target_slices,
        seed=args.seed,
        weighted_sampling=True,
    )

    # Model
    model = build_3d_model(in_channels=1, num_classes=1).to(device)

    # Loss (class imbalance from TRAIN jsonl)
    pos_w = pos_weight_from_jsonl(args.train).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_dir=args.out,
        
    )