import os
import random
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pos_weight_from_jsonl(jsonl_path: str) -> torch.Tensor:
    """
    Calculate the positive class weight for BCEWithLogitsLoss.
    The formula is neg_count / pos_count.
    """
    labels = [int(json.loads(l)["label"]) for l in Path(jsonl_path).read_text().splitlines()]
    pos = sum(labels)
    neg = len(labels) - pos
    weight = neg / max(pos, 1)
    return torch.tensor([weight], dtype=torch.float32)


def build_binary_metrics():
    return MetricCollection({
        "acc": BinaryAccuracy(),
        "auroc": BinaryAUROC(),
        "ap": BinaryAveragePrecision(),
    })


def save_checkpoint(path: Path, 
                    model_state: dict, 
                    optimizer_state: dict, 
                    epoch: Optional[int] = None,
) -> None:
    dest = Path(path)
    
    payload = {"model": model_state}
    if optimizer_state:
        payload["optimizer"] = optimizer_state
    if epoch is not None:
        payload["epoch"] = epoch

    tmp = (
        dest.with_suffix(dest.suffix + ".tmp")
        if dest.suffix
        else dest.with_suffix(".tmp")
    )
    torch.save(payload, tmp)
    os.replace(tmp, dest)


