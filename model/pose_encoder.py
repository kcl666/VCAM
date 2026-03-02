import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    """
    Ecode relative camera pose into token embedding
    """
    def __init__(self, pose_dim=6, out_dim=256, hidden_dim=128):
        super().__init__()

        self.pose_dim = pose_dim
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, rel_pose):
        """
        rel_pose: (B, V, pose_dim)
        return:   (B, V, out_dim)
        """
        B, V, D = rel_pose.shape
        assert D == self.pose_dim, f"Excepted pose_dim={self.pose_dim}, got {D}"

        rel_pose = rel_pose.view(B * V, D)
        pose_token = self.mlp(rel_pose)
        pose_token = pose_token.view(B, V, -1)
        return pose_token
