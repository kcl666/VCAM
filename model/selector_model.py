import torch
import torch.nn as nn

from models.image_encoder import ImageFeatureEncoder
from models.pose_encoder import PoseEncoder
from models.token_fusion import TokenFusion
from models.selector_transformer import SelectorTransformer


class SelectorModel(nn.Module):
    """
    View selector model.

    Input:
        images:    (B, V, 3, H, W)
        rel_poses: (B, V, 6)

    Output:
        delta_pred: (B, V)
    """

    def __init__(
        self,
        args,
        embed_dim: int = 256,
        backbone: str = "resnet50",
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ===== Encoders =====
        self.image_encoder = ImageFeatureEncoder(
            backbone=backbone,
            out_dim=embed_dim,
        )

        self.pose_encoder = PoseEncoder(
            out_dim=embed_dim,
        )

        # ===== Token Fusion =====
        self.token_fusion = TokenFusion(
            embed_dim=embed_dim,
            method=args.fusion_mode,  # 可后续 修改
        )

        # ===== Transformer =====
        self.transformer = SelectorTransformer(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ===== Score Head =====
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        rel_poses: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ):
        """
        images:    (B, V, 3, H, W)
        rel_poses: (B, V, 6)
        view_mask: (B, V), optional, True = ignore

        return:
            delta_pred: (B, V)
        """

        # ===== Encode =====
        img_tokens = self.image_encoder(images)        # (B, V, D)
        pose_tokens = self.pose_encoder(rel_poses)     # (B, V, D)

        # ===== Fuse =====
        tokens = self.token_fusion(
            img_tokens,
            pose_tokens,
        )  # (B, V, D)

        # ===== Transformer =====
        tokens = self.transformer(
            tokens,
            attn_mask=view_mask,
        )  # (B, V, D)

        # ===== Regression =====
        delta_pred = self.score_head(tokens).squeeze(-1)  # (B, V)

        return delta_pred
