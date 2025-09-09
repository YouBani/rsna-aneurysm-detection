import torch.nn as nn
from torchvision.models.video import r3d_18

def build_3d_model(
        in_channels: int=1,
        num_classes: int=1,
) -> nn.Module:
    """
    3D ResNet-18 baseline for (C, Z, H, W) inputs.
    Returns raw logits (no sigmoid).
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
    return model