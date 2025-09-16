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


class MultiHead(nn.Module):
    """
    Outputs: {"presence": (B,), "locations": (B, K)}
    """

    def __init__(self, in_features: int, num_locations: int):
        super().__init__()
        self.presence = nn.Linear(in_features, 1)
        self.num_locations = int(num_locations)
        self.locations = (
            nn.Linear(in_features, self.num_locations)
            if self.num_locations > 0
            else None
        )

    def forward(self, x):
        out = {"presence": self.presence(x).squeeze(1)}
        if self.locations is not None:
            out["locations"] = self.locations(x)
        return out


def build_3d_model(
    in_channels: int = 1,
    num_classes: int = 1,
    checkpointing: bool = False,
    use_groupnorm: bool = False,
    num_locations: int = 0,
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
    # Replace the classifier with a single logit head
    in_feats = model.fc.in_features
    if num_locations > 0:
        model.fc = MultiHead(in_feats, num_locations)  # type: ignore
    else:
        model.fc = nn.Linear(in_feats, num_classes)

    if checkpointing:
        print("Gradient checkpointing enabled for ResNet layers.")
        model.layer1 = CheckpointedSeq(model.layer1)  # type: ignore
        model.layer2 = CheckpointedSeq(model.layer2)  # type: ignore
        model.layer3 = CheckpointedSeq(model.layer3)  # type: ignore
        model.layer4 = CheckpointedSeq(model.layer4)  # type: ignore

    return model
