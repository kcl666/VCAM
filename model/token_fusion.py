import torch
import torch.nn as nn


class TokenFusion(nn.Module):
    """
    Fuse image token and pose tokens into a unified token representation

    Iuput:
        img_tokens : (B, N, D)
        pose_tokens : (B, N, D)

    Output:
        fused_tokens : (B, N, D)

    """
    def __init__(
            self,
            embed_dim,
            method="add",
            dropout=0.0,
    ):
        """
        Args:
            embed_dim (int): token embedding dimension D
            method (str): fusion method
                - "add"
                - "concat"
                - "gated"
                - "film"
            dropout (float): optional dropout after fusion
        """
        super().__init__()
        self.dim = embed_dim
        self.method = method
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if method == "add":
            # No Learnable parameters needed
            self.fuse = None

        elif method == "concat":
            # [img || pose] -> D
            self.fuse = nn.Linear(embed_dim * 2, embed_dim)

        elif method == "gated":
            # g = sigmoid(w * pose)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )

        elif method == "film":
            # Generate scale (game) and shift (beta)
            self.film = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2)
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, img_tokens, pose_tokens):
        """
        Args:
            img_tokens: (B, N, D)
            pose_tokens: (B, N, D)
        """
        assert img_tokens.shape == pose_tokens.shape, \
            "Image tokens and pose tokens must have same shape"

        if self.method == "add":
            fused = img_tokens + pose_tokens

        elif self.method == "concat":
            fused = torch.cat((img_tokens, pose_tokens), dim=-1)
            fused = self.fuse(fused)

        elif self.method == "gated":
            gate = self.gate(pose_tokens)
            fused = img_tokens + pose_tokens * gate

        elif self.method == "film":
            gamma_beta = self.film(pose_tokens)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            fused = gamma * img_tokens + beta

        else:
            raise RuntimeError(f"Invalid fusion method")

        fused = self.dropout(fused)
        return fused