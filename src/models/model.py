import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.checkpoint import checkpoint_sequential


def replace_bn_with_gn(module: nn.Module, num_groups: int = 16):
    """Recursively find all BatchNorm3d layers and replace them with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            num_channels = child.num_features
            setattr(
                module,
                name,
                nn.GroupNorm(
                    num_groups=min(num_groups, num_channels), num_channels=num_channels
                ),
            )
        else:
            replace_bn_with_gn(child, num_groups)


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
    num_outputs: int = 14,
    checkpointing: bool = False,
    use_groupnorm: bool = False,
) -> nn.Module:
    """
    3D ResNet-18 baseline for (C, Z, H, W) inputs.
    Returns raw logits (no sigmoid).
    Optionally enables gradient checkpointing on layer1 to 4.
    """
    model = r3d_18(weights=None)

    if use_groupnorm:
        print("Replacing BatchNorm with GroupNorm.")
        replace_bn_with_gn(model)

    old = model.stem[0]  # type: ignore
    model.stem[0] = nn.Conv3d(  # type: ignore
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_outputs)  # type: ignore

    if checkpointing:
        print("Gradient checkpointing enabled for ResNet layers.")
        model.layer1 = CheckpointedSeq(model.layer1)  # type: ignore
        model.layer2 = CheckpointedSeq(model.layer2)  # type: ignore
        model.layer3 = CheckpointedSeq(model.layer3)  # type: ignore
        model.layer4 = CheckpointedSeq(model.layer4)  # type: ignore

    return model
