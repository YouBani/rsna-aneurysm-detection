import torch
import torch.nn as nn
import timm


class SliceModel25D(nn.Module):
    """
    2.5D slice-level classifier.
    Input: (B, 3, H, W)  where channels = [z-1, z, z+1]
    Output: (B, 1)  aneurysm slice probability (logits)
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_ch: int = 3,
        drop: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_ch,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feat_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        logits = self.head(feat)
        return logits
