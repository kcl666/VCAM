import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, features):
        """
        features: list of feature maps [C2, C3, C4, C5]
        """
        last_inner = self.lateral_convs[-1](features[-1])
        outs = [self.output_convs[-1](last_inner)]

        for idx in range(len(features) - 2, -1, -1):
            feat = features[idx]
            lateral = self.lateral_convs[idx](feat)
            upsampled = F.interpolate(
                last_inner, size=lateral.shape[-2:], mode="nearest"
            )
            last_inner = lateral + upsampled
            outs.insert(0, self.output_convs[idx](last_inner))

        return outs


class ImageFeatureEncoder(nn.Module):
    """
    Encode each view image into a compact token vector.
    """

    def __init__(
        self,
        backbone="resnet18",
        pretrained=True,
        fpn_dim=256,
        out_dim=256,
    ):
        super().__init__()

        # ===== Backbone =====
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_channels_list = [64, 128, 256, 512]
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Stem
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )

        # Stages
        self.layer1 = net.layer1  # C2
        self.layer2 = net.layer2  # C3
        self.layer3 = net.layer3  # C4
        self.layer4 = net.layer4  # C5

        # ===== FPN =====
        self.fpn = FPN(
            in_channels_list=in_channels_list,
            out_channels=fpn_dim,
        )

        # ===== Token Projection =====
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(fpn_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        """
        images: (B, V, 3, H, W)
        return: (B, V, out_dim)
        """
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        # Backbone forward
        x = self.stem(images)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN fusion
        fpn_feats = self.fpn([c2, c3, c4, c5])

        # Global pooling for each scale
        pooled = []
        for feat in fpn_feats:
            p = self.pool(feat).flatten(1)
            pooled.append(p)

        # Concatenate multi-scale info
        fused = torch.cat(pooled, dim=1)

        # Token projection
        token = self.fc(fused)
        token = token.view(B, V, -1)

        return token
