import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.checkpoint import checkpoint_sequential


class CheckpointedSeq(nn.Module):
    """Wraps an nn.Sequential with gradient checkpointing."""

    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.seq(x)

        return checkpoint_sequential(self.seq, len(self.seq), x, use_reentrant=False)


def build_3d_model(
    in_channels: int = 1,
    num_classes: int = 1,
    checkpointing: bool = False,
) -> nn.Module:
    """
    3D ResNet-18 baseline for (C, Z, H, W) inputs.
    Returns raw logits (no sigmoid).
    Optionally enables gradient checkpointing on layer1 to 4.
    """
    model = r3d_18(weights=None)
    old = model.stem[0]
    model.stem[0] = nn.Conv3d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    # Replace the classifier with a single logit head
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    if checkpointing:
        print("Gradient checkpointing enabled for ResNet layers.")
        model.layer1 = CheckpointedSeq(model.layer1)
        model.layer2 = CheckpointedSeq(model.layer2)
        model.layer3 = CheckpointedSeq(model.layer3)
        model.layer4 = CheckpointedSeq(model.layer4)

    return model
