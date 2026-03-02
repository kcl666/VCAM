import torch
import torch.nn as nn
from typing import Optional


class SelectorTransformer(nn.Module):
    """
    Transformer encoder for view-level token interaction.

    Input:
        tokens: (B, v, D)
        attn_mask: optional (B, v) or (V, V)

    Output:
        tokens: (B, V, D)
    """

    def __init__(
            self,
            embed_dim: int = 256,
            num_layers: int = 3,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            tokens: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        tokens: (B, v, D)
        attn_mask: optional
            - (V, V): shared attention mask
            - (B, v): key padding mask (True = ignore)
        return:
            encoded tokens: (B, V, D)
        """

        if attn_mask is not None:
            # PyTorch Transformer experts key_padding_mask as (B, V)
            if attn_mask.dim() == 2 or attn_mask.shape[0] == tokens.shape[0]:
                out = self.encoder(tokens, src_key_padding_mask=attn_mask)
            else:
                out = self.encoder(tokens, mask=attn_mask)
        else:
            out = self.encoder(tokens)

        out = self.norm(out)
        return out