import torch
from torchmetrics.classification import MultilabelAUROC
from src.constants.rsna import K, PRESENT_IDX


def build_multilabel_auroc() -> MultilabelAUROC:
    """Returns a torchmetrics MultilabelAUROC that outputs per-label AUCs."""
    return MultilabelAUROC(num_labels=K, average=None)


def weighted_multilabel_auc(per_label_auc: torch.Tensor) -> torch.Tensor:
    """
    Returns weighted column-wise AUC.
    """
    per_label_auc = torch.nan_to_num(per_label_auc, nan=0.5)
    w = torch.ones_like(per_label_auc)
    w[PRESENT_IDX] = 13.0
    return (per_label_auc * w).sum() / w.sum()
